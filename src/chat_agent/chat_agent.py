"""
chat_agent — multi-user chatbot with 3-tier hybrid memory.

Tier 1: in-process ShortTermBuffer (per-conversation deque)
Tier 2: SQLite (durable raw history, write-through)
Tier 3: Mem0 (extracted long-term facts, async fire-and-forget)

Run:  uvicorn chat_agent:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import sys
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import AsyncIterator, Literal, Protocol

import aiosqlite
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()  # so OPENAI_API_KEY in .env reaches Mem0's OpenAI client

# Make the sibling modules importable: intent_classifier (project root),
# safety package (project root), and portfolio_check package (its own dir).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "portfolio_check"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

log = logging.getLogger("chat_agent")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_path: str = "chat.db"
    max_turns_per_conv: int = 20
    max_active_convs: int = 1000
    memory_top_k: int = 5
    mem0_enabled: bool = True


settings = Settings()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

Role = Literal["user", "assistant", "system", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    id: int | None = None
    conversation_id: str | None = None
    user_id: str | None = None
    created_at: datetime | None = None


@dataclass
class Memory:
    text: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tier 2: SQLite store
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    external_id TEXT UNIQUE,
    display_name TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
    ON conversations(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user','assistant','system','tool')),
    content TEXT NOT NULL,
    meta_json TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_messages_user_created ON messages(user_id, created_at DESC);
"""


def _new_id() -> str:
    return uuid.uuid4().hex


def _row_to_message(r: aiosqlite.Row) -> Message:
    ts = r["created_at"]
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            ts = None
    return Message(
        id=r["id"],
        conversation_id=r["conversation_id"],
        user_id=r["user_id"],
        role=r["role"],
        content=r["content"],
        created_at=ts,
    )


class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @asynccontextmanager
    async def _conn(self):
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            yield db

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()

    async def upsert_user(
        self, user_id: str, external_id: str | None = None, display_name: str | None = None
    ) -> None:
        async with self._conn() as db:
            await db.execute(
                "INSERT OR IGNORE INTO users (id, external_id, display_name) VALUES (?, ?, ?)",
                (user_id, external_id, display_name),
            )
            await db.commit()

    async def create_conversation(self, user_id: str, title: str | None = None) -> str:
        cid = _new_id()
        async with self._conn() as db:
            await db.execute(
                "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
                (cid, user_id, title),
            )
            await db.commit()
        return cid

    async def get_conversation(self, conversation_id: str) -> dict | None:
        async with self._conn() as db:
            cur = await db.execute(
                "SELECT id, user_id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]:
        async with self._conn() as db:
            cur = await db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [dict(r) for r in await cur.fetchall()]

    async def append_message(
        self,
        conversation_id: str,
        user_id: str,
        role: Role,
        content: str,
        meta: dict | None = None,
    ) -> Message:
        meta_json = json.dumps(meta) if meta else None
        async with self._conn() as db:
            cur = await db.execute(
                "INSERT INTO messages (conversation_id, user_id, role, content, meta_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, user_id, role, content, meta_json),
            )
            mid = cur.lastrowid
            await db.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conversation_id,),
            )
            await db.commit()
            cur = await db.execute(
                "SELECT id, conversation_id, user_id, role, content, created_at "
                "FROM messages WHERE id = ?",
                (mid,),
            )
            row = await cur.fetchone()
        return _row_to_message(row)

    async def get_recent_messages(self, conversation_id: str, limit: int) -> list[Message]:
        async with self._conn() as db:
            cur = await db.execute(
                "SELECT id, conversation_id, user_id, role, content, created_at "
                "FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            )
            rows = await cur.fetchall()
        msgs = [_row_to_message(r) for r in rows]
        msgs.reverse()
        return msgs


# ---------------------------------------------------------------------------
# Tier 1: ShortTermBuffer
# ---------------------------------------------------------------------------

class ConversationBuffer:
    __slots__ = ("messages", "last_access", "hydrated", "lock")

    def __init__(self, maxlen: int):
        self.messages: deque[Message] = deque(maxlen=maxlen)
        self.last_access: float = time.monotonic()
        self.hydrated: bool = False
        self.lock: asyncio.Lock = asyncio.Lock()


class ShortTermBuffer:
    def __init__(self, max_turns_per_conv: int, max_active_convs: int):
        self.max_turns = max_turns_per_conv
        self.max_active = max_active_convs
        self._buffers: OrderedDict[str, ConversationBuffer] = OrderedDict()
        self._meta_lock = asyncio.Lock()

    async def get(self, conversation_id: str) -> ConversationBuffer:
        async with self._meta_lock:
            buf = self._buffers.get(conversation_id)
            if buf is None:
                buf = ConversationBuffer(self.max_turns)
                self._buffers[conversation_id] = buf
                while len(self._buffers) > self.max_active:
                    self._buffers.popitem(last=False)
            else:
                self._buffers.move_to_end(conversation_id)
            buf.last_access = time.monotonic()
            return buf

    async def invalidate(self, conversation_id: str) -> None:
        async with self._meta_lock:
            self._buffers.pop(conversation_id, None)


# ---------------------------------------------------------------------------
# Tier 3: Memory store (Mem0 wrapper + Noop fallback)
# ---------------------------------------------------------------------------

class MemoryStore(Protocol):
    async def search(self, query: str, user_id: str, top_k: int = 5) -> list[Memory]: ...
    async def add(
        self, messages: list[Message], user_id: str, metadata: dict | None = None
    ) -> None: ...


class NoopMemoryStore:
    async def search(self, query: str, user_id: str, top_k: int = 5) -> list[Memory]:
        return []

    async def add(
        self, messages: list[Message], user_id: str, metadata: dict | None = None
    ) -> None:
        return None


class Mem0Store:
    """Thin async wrapper around Mem0 OSS. Lazy-imports mem0 so the dep stays optional."""

    # Pin to gpt-4o-mini: Mem0 2.0.1's default (gpt-5-mini) sends `max_tokens` to
    # OpenAI, which reasoning models reject — they want `max_completion_tokens`.
    DEFAULT_CONFIG = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"},
        },
    }

    def __init__(self, config: dict | None = None):
        from mem0 import Memory as _Mem0  # noqa: WPS433

        self._mem = _Mem0.from_config(config or self.DEFAULT_CONFIG)

    async def search(self, query: str, user_id: str, top_k: int = 5) -> list[Memory]:
        try:
            res = await asyncio.to_thread(
                self._mem.search,
                query=query,
                filters={"user_id": user_id},
                top_k=top_k,
            )
            items = res.get("results", res) if isinstance(res, dict) else res
            return [
                Memory(
                    text=item.get("memory") or item.get("text") or "",
                    score=float(item.get("score", 0.0)),
                    metadata=item.get("metadata") or {},
                )
                for item in (items or [])
            ]
        except Exception as e:
            log.warning("mem0 search failed: %s", e)
            return []

    async def add(
        self, messages: list[Message], user_id: str, metadata: dict | None = None
    ) -> None:
        try:
            payload = [{"role": m.role, "content": m.content} for m in messages]
            await asyncio.to_thread(
                self._mem.add, messages=payload, user_id=user_id, metadata=metadata or {}
            )
        except Exception as e:
            log.warning("mem0 add failed: %s", e)


# ---------------------------------------------------------------------------
# Agent (stub returns random hex; real agent drops in here later)
# ---------------------------------------------------------------------------

class Agent(Protocol):
    def astream(
        self,
        messages: list[Message],
        memories: list[Memory],
        user_id: str,
        conversation_id: str,
    ) -> AsyncIterator[str]: ...


class StubAgent:
    """Streams a random hex token in 4-char chunks with small delays."""

    async def astream(
        self,
        messages: list[Message],
        memories: list[Memory],
        user_id: str,
        conversation_id: str,
    ) -> AsyncIterator[str]:
        token = secrets.token_hex(8)
        for i in range(0, len(token), 4):
            await asyncio.sleep(0.05)
            yield token[i : i + 4]


# ---------------------------------------------------------------------------
# Safety guardrail wrapper
# ---------------------------------------------------------------------------

class FinancialSafety:
    """Lazy wrapper around safety.inference.FinancialGuardrail.

    The ONNX model is heavy; load on first use so app boot stays fast and
    test environments without the artefacts can still import this module.
    """

    def __init__(self) -> None:
        self._guardrail = None
        self._compose_reason = None
        self._load_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._guardrail is not None:
            return
        async with self._load_lock:
            if self._guardrail is not None:
                return
            from safety.inference import FinancialGuardrail, compose_reason  # noqa: WPS433
            self._guardrail = await asyncio.to_thread(FinancialGuardrail)
            self._compose_reason = compose_reason

    async def check(self, query: str) -> dict:
        try:
            await self._ensure_loaded()
            verdict = await asyncio.to_thread(self._guardrail.classify, query)
            return verdict
        except Exception as e:  # noqa: BLE001
            log.warning("safety check failed, allowing by default: %s", e)
            return {"query": query, "should_block": False, "category": "general"}


# ---------------------------------------------------------------------------
# Users DB (lookup user_context from portfolio_check/users.json)
# ---------------------------------------------------------------------------

class UsersDB:
    def __init__(self, path: Path):
        self.path = path
        self._users: dict | None = None

    def _load(self) -> dict:
        if self._users is None:
            try:
                self._users = json.loads(self.path.read_text())
            except Exception as e:  # noqa: BLE001
                log.warning("could not load users.json at %s: %s", self.path, e)
                self._users = {}
        return self._users

    def get(self, user_id: str) -> dict:
        users = self._load()
        ctx = users.get(user_id)
        if ctx is None:
            return {"user_id": user_id}
        return ctx


# ---------------------------------------------------------------------------
# Orchestrator agent: safety -> intent classify -> route
# ---------------------------------------------------------------------------

WIRED_INTENTS = {"portfolio_health", "general_query"}

ALL_INTENTS = {
    "portfolio_health", "market_research", "investment_strategy",
    "financial_planning", "financial_calculator", "risk_assessment",
    "product_recommendation", "predictive_analysis",
    "customer_support", "general_query",
}


def _json_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"not serializable: {type(o)}")


class OrchestratorAgent:
    """Implements the Agent protocol; replaces StubAgent in the pipeline.

    Per turn:
      1. Run safety guardrail on the user's text.
         -> blocked: yield a single JSON chunk {blocked, category, reason} and stop.
      2. Look up user_context from users.json by user_id.
      3. Call intent_classifier(user_context, text).
      4. Dispatch:
           portfolio_health -> run_portfolio_check(user_context); yield JSON
           general_query    -> gpt-4o-mini chat completion stream over recent messages
           anything else    -> yield {"status": "agent_not_built", ...}
    """

    GENERAL_MODEL = "gpt-4o-mini"

    def __init__(self, safety: FinancialSafety, users_db: UsersDB):
        self.safety = safety
        self.users_db = users_db
        self._openai = AsyncOpenAI()

    async def astream(
        self,
        messages: list[Message],
        memories: list[Memory],
        user_id: str,
        conversation_id: str,
    ) -> AsyncIterator[str]:
        # The most recent user message is what we classify / route on.
        text = messages[-1].content if messages else ""

        # 1. Safety
        verdict = await self.safety.check(text)
        if verdict.get("should_block"):
            category = verdict.get("category", "unknown")
            reason = (
                f"This query was blocked by the financial safety guardrail "
                f"(category: {category}). It reads as an operational request "
                f"for activity that we cannot assist with."
            )
            yield json.dumps(
                {"blocked": True, "category": category, "reason": reason}
            )
            return

        # 2. User context lookup
        user_context = self.users_db.get(user_id)

        # 3. Intent classification (sync; offload)
        from intent_classifier import intent_classifier  # noqa: WPS433
        try:
            user_ctx_str = json.dumps(user_context, default=_json_default)
        except Exception:
            user_ctx_str = json.dumps({"user_id": user_id})
        intent_out = await asyncio.to_thread(
            intent_classifier, user_ctx_str, text
        )
        agent_name = (intent_out or {}).get("agent", "general_query")

        # 4. Dispatch
        if agent_name == "portfolio_health":
            async for chunk in self._run_portfolio_health(user_context):
                yield chunk
        elif agent_name == "general_query":
            async for chunk in self._run_general_query(messages, memories):
                yield chunk
        elif agent_name in ALL_INTENTS:
            yield json.dumps(
                {
                    "status": "agent_not_built",
                    "intent": agent_name,
                    "entities": (intent_out or {}).get("entities", {}),
                    "message": (
                        f"No agent is wired for intent '{agent_name}' yet."
                    ),
                }
            )
        else:
            yield json.dumps(
                {
                    "status": "agent_not_built",
                    "intent": agent_name,
                    "message": "Unknown intent label.",
                }
            )

    # --- portfolio_health --------------------------------------------------

    async def _run_portfolio_health(self, user_context: dict) -> AsyncIterator[str]:
        try:
            from portfolio_check.agent import run_portfolio_check  # noqa: WPS433
            result = await run_portfolio_check(user_context)
            yield json.dumps(result, default=_json_default)
        except Exception as e:  # noqa: BLE001
            log.exception("portfolio_check failed")
            yield json.dumps(
                {
                    "status": "agent_error",
                    "intent": "portfolio_health",
                    "error": str(e),
                }
            )

    # --- general_query (gpt-4o-mini) --------------------------------------

    async def _run_general_query(
        self, messages: list[Message], memories: list[Memory]
    ) -> AsyncIterator[str]:
        memory_block = ""
        if memories:
            memory_block = "\n\nRelevant prior memory about this user:\n" + "\n".join(
                f"- {m.text}" for m in memories if m.text
            )
        system = (
            "You are a helpful financial assistant. Be concise and clear. "
            "If the question needs personal financial data you do not have, "
            "say what would be needed rather than inventing it."
            + memory_block
        )

        oa_messages: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role in ("user", "assistant"):
                oa_messages.append({"role": m.role, "content": m.content})

        try:
            stream = await self._openai.chat.completions.create(
                model=self.GENERAL_MODEL,
                messages=oa_messages,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:  # noqa: BLE001
            log.exception("general_query LLM call failed")
            yield json.dumps(
                {"status": "agent_error", "intent": "general_query", "error": str(e)}
            )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Orchestrator (per-turn pipeline)
# ---------------------------------------------------------------------------

class ChatService:
    def __init__(
        self,
        sqlite: SQLiteStore,
        buffer: ShortTermBuffer,
        memory: MemoryStore,
        agent: Agent,
        memory_top_k: int = 5,
    ):
        self.sqlite = sqlite
        self.buffer = buffer
        self.memory = memory
        self.agent = agent
        self.memory_top_k = memory_top_k

    async def _hydrate(self, buf: ConversationBuffer, conversation_id: str) -> None:
        if buf.hydrated:
            return
        msgs = await self.sqlite.get_recent_messages(conversation_id, self.buffer.max_turns)
        buf.messages.clear()
        for m in msgs:
            buf.messages.append(m)
        buf.hydrated = True

    async def stream_turn(
        self, user_id: str, conversation_id: str, text: str
    ) -> AsyncIterator[str]:
        """Per-turn pipeline as an SSE event stream."""
        buf = await self.buffer.get(conversation_id)
        user_msg = None
        assistant_msg = None
        try:
            async with buf.lock:
                await self._hydrate(buf, conversation_id)

                user_msg = await self.sqlite.append_message(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    role="user",
                    content=text,
                )
                buf.messages.append(user_msg)

                recent = list(buf.messages)
                memories = await self.memory.search(
                    query=text, user_id=user_id, top_k=self.memory_top_k
                )

                yield _sse("start", {"message_id": user_msg.id, "memories_used": len(memories)})

                chunks: list[str] = []
                async for delta in self.agent.astream(
                    messages=recent,
                    memories=memories,
                    user_id=user_id,
                    conversation_id=conversation_id,
                ):
                    chunks.append(delta)
                    yield _sse("chunk", {"text": delta})

                full = "".join(chunks)
                assistant_msg = await self.sqlite.append_message(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    role="assistant",
                    content=full,
                )
                buf.messages.append(assistant_msg)
        except Exception as e:  # noqa: BLE001
            log.exception("stream_turn failed")
            yield _sse("error", {"message": str(e)})
            return

        asyncio.create_task(
            self.memory.add(
                messages=[user_msg, assistant_msg],
                user_id=user_id,
                metadata={"conversation_id": conversation_id},
            )
        )

        yield _sse(
            "done",
            {
                "message_id": assistant_msg.id,
                "content": assistant_msg.content,
                "buffer_size": len(buf.messages),
            },
        )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

class CreateUserBody(BaseModel):
    user_id: str
    external_id: str | None = None
    display_name: str | None = None


class CreateConversationBody(BaseModel):
    user_id: str
    title: str | None = None


class ChatBody(BaseModel):
    user_id: str
    conversation_id: str
    message: str = Field(min_length=1)


def _build_memory(enabled: bool) -> "MemoryStore":
    if not enabled:
        log.info("Mem0 disabled by config; using NoopMemoryStore")
        return NoopMemoryStore()
    try:
        store = Mem0Store()
        log.info("Mem0 initialized — long-term memory active")
        return store
    except Exception as e:  # noqa: BLE001
        log.error(
            "Mem0 init failed (%s). Install `mem0ai` and set OPENAI_API_KEY (or "
            "configure another provider). Falling back to NoopMemoryStore.",
            e,
        )
        return NoopMemoryStore()


def build_app() -> FastAPI:
    sqlite = SQLiteStore(settings.db_path)
    buffer = ShortTermBuffer(settings.max_turns_per_conv, settings.max_active_convs)
    memory: MemoryStore = _build_memory(settings.mem0_enabled)

    users_db = UsersDB(_PROJECT_ROOT / "portfolio_check" / "portfolio_check" / "users.json")
    safety = FinancialSafety()
    agent: Agent = OrchestratorAgent(safety=safety, users_db=users_db)

    service = ChatService(
        sqlite=sqlite,
        buffer=buffer,
        memory=memory,
        agent=agent,
        memory_top_k=settings.memory_top_k,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await sqlite.init()
        yield

    app = FastAPI(title="chat_agent", lifespan=lifespan)

    @app.post("/users")
    async def create_user(body: CreateUserBody):
        await sqlite.upsert_user(body.user_id, body.external_id, body.display_name)
        return {"user_id": body.user_id}

    @app.post("/conversations")
    async def create_conversation(body: CreateConversationBody):
        cid = await sqlite.create_conversation(body.user_id, body.title)
        return {"conversation_id": cid}

    @app.get("/conversations")
    async def list_conversations(user_id: str, limit: int = 50):
        return await sqlite.list_conversations(user_id, limit)

    @app.get("/conversations/{conversation_id}/messages")
    async def list_messages(conversation_id: str, limit: int = 100):
        msgs = await sqlite.get_recent_messages(conversation_id, limit)
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in msgs
        ]

    @app.post("/chat")
    async def chat(body: ChatBody):
        conv = await sqlite.get_conversation(body.conversation_id)
        if conv is None:
            raise HTTPException(404, "conversation not found")
        if conv["user_id"] != body.user_id:
            raise HTTPException(403, "user does not own this conversation")
        return StreamingResponse(
            service.stream_turn(body.user_id, body.conversation_id, body.message),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = build_app()
