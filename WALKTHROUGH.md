# Walkthrough — How to run, and how the code is laid out

This is the operator's guide. The README explains *why* each piece exists; this file explains *where it lives* and *how to drive it*.

---

## 1. Prerequisites

| | |
|---|---|
| Python | 3.11+ |
| OS | macOS / Linux (tested), Windows works for the service; training scripts assume `torch.backends.mps` on Apple silicon but fall back to CPU on Windows |
| Optional | `uvx` (for the yfinance MCP server) — `pip install uv` if missing |
| Required env | `OPENAI_API_KEY` (only at request time; tests do not need it) |

---

## 2. Setup

```bash
git clone <repo-url>
cd valura-ai-ai-engineer-assignment-PrabhasKalyan-main

python -m venv venv
source venv/bin/activate            # Linux/macOS
# venv\Scripts\activate             # Windows

pip install -r requirements.txt

cp .env.example .env
# Edit .env — set OPENAI_API_KEY=sk-...
```

The first time the service handles a `/chat` request it will load the quantized BERT-mini ONNX model from `src/safety/bert_onnx/`. That artefact is checked into the repo, so you do **not** need to retrain.

---

## 3. Run the service

```bash
uvicorn src.chat_agent.chat_agent:app --reload --host 0.0.0.0 --port 8000
```

You should see:

```
INFO  Mem0 initialized — long-term memory active        # if mem0ai installs
INFO  Application startup complete.
INFO  Uvicorn running on http://0.0.0.0:8000
```

If Mem0 fails to init (no API key, network), you'll see:
```
ERROR Mem0 init failed (...). Falling back to NoopMemoryStore.
```
The service still works — Tiers 1 (in-memory buffer) and 2 (SQLite) carry the load.

---

## 4. Drive it end-to-end

Three calls to take a fresh user from zero to a streamed answer:

```bash
# 1. Register a user (use one of the fixture user_ids so portfolio_health has data)
curl -s -X POST http://localhost:8000/users \
  -H 'content-type: application/json' \
  -d '{"user_id": "user_003_concentrated", "display_name": "Demo"}'

# 2. Open a conversation
CID=$(curl -s -X POST http://localhost:8000/conversations \
  -H 'content-type: application/json' \
  -d '{"user_id": "user_003_concentrated", "title": "demo"}' | jq -r .conversation_id)

# 3. Chat — note the SSE stream (-N disables curl's buffering)
curl -N -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d "{\"user_id\": \"user_003_concentrated\", \"conversation_id\": \"$CID\", \"message\": \"how is my portfolio doing?\"}"
```

Expected SSE events:
```
event: start
data: {"message_id": 1, "memories_used": 0}

event: chunk
data: {"text": "{\"user_id\": \"user_003_concentrated\", ... }"}

event: done
data: {"message_id": 2, "content": "...", "buffer_size": 2}
```

### Try the interesting paths

| Query | Path it exercises |
|---|---|
| `"What is insider trading?"` | safety → **edu**, allowed → classifier → `general_query` → streamed answer |
| `"How do I do insider trading?"` | safety → **op**, **blocked** → single chunk, classifier never runs |
| `"how is my portfolio doing?"` (user_003_concentrated) | classifier → `portfolio_health` → yfinance MCP → metrics → observations |
| same query with `user_004_empty` | classifier → `portfolio_health` → BUILD path (no MCP) |
| `"forecast NVDA next year"` | classifier → `predictive_analysis` → **stub** `agent_not_built` |

---

## 5. Run the tests

```bash
pytest tests/ -v
```

`OPENAI_API_KEY` is **not** required — `tests/conftest.py` mocks the OpenAI client. The three suites:

| File | What it checks |
|---|---|
| `tests/test_safety_pairs.py` | Recall ≥ 95% on harmful queries, pass-through ≥ 90% on educational ones (fixtures: `safety_pairs.json`) |
| `tests/test_classifier_routing.py` | Routing accuracy ≥ 85% with subset entity matching (fixtures: `intent_classification.json`) |
| `tests/test_portfolio_health_skeleton.py` | Agent runs on every fixture user incl. `user_004_empty` without crashing |

---

## 6. Code map — directory by directory

```
.
├── README.md                          ← architecture & decisions
├── WALKTHROUGH.md                     ← this file
├── ASSIGNMENT.md                      ← original spec
├── requirements.txt
├── pytest.ini
├── fixtures/                          ← graded gold files (do not edit)
│   ├── users/                         ← 5 user profiles (user_001..user_008)
│   ├── conversations/                 ← follow-up / topic-switch / multi-intent transcripts
│   └── test_queries/
│       ├── intent_classification.json ← classifier gold set
│       └── safety_pairs.json          ← safety gold set (harmful + educational pairs)
├── src/
│   ├── intent_classifier.py           ← (1) single LLM call + TF-IDF fallback
│   ├── chat_agent/                    ← (HTTP) FastAPI app, orchestrator, 3-tier memory
│   │   ├── chat_agent.py              ← all of it lives here, by design
│   │   ├── chat.db                    ← SQLite Tier-2 store (auto-created)
│   │   └── pyproject.toml
│   ├── safety/                        ← (2) BERT-mini guardrail
│   │   ├── inference.py               ← runtime: load ONNX, classify, compose reason
│   │   ├── build_dataset_bert.py      ← raw HF datasets → 13-class jsonl
│   │   ├── train_bert.py              ← fine-tune prajjwal1/bert-mini
│   │   ├── export_onnx.py             ← ONNX + dynamic INT8 quantization
│   │   ├── train_and_export.py        ← convenience wrapper over the two above
│   │   ├── eval_bert.py               ← report on test.jsonl
│   │   ├── raw/                       ← PKU-Beaver, AgentHarm, etc.
│   │   ├── data_bert/                 ← train/val/test jsonl + labels.json
│   │   ├── bert_ckpt/                 ← Trainer checkpoints (post-training)
│   │   └── bert_onnx/                 ← ✱ the production artefact ✱
│   │       └── model_quantized.onnx
│   └── portfolio_check/               ← (3) Portfolio Health agent
│       └── portfolio_check/
│           ├── agent.py               ← orchestration: validate → MCP → metrics → LLM-prose
│           ├── metrics.py             ← deterministic math (no LLM)
│           ├── mcp_client.py          ← yfinance MCP stdio client
│           ├── local_yfinance.py      ← fallback when PORTFOLIO_DATA_BACKEND=local
│           ├── users.json             ← demo users DB (prod: Postgres/Mongo)
│           └── __main__.py            ← `python -m portfolio_check <user_id>`
└── tests/
    ├── conftest.py                    ← mocks OpenAI; loads fixtures
    ├── test_safety_pairs.py
    ├── test_classifier_routing.py
    └── test_portfolio_health_skeleton.py
```

---

## 7. Code walkthrough — follow one request through the system

The whole service is one orchestrator and four subsystems. Read in this order:

### Step A — request hits the HTTP layer
**File:** `src/chat_agent/chat_agent.py` → `build_app()` → `@app.post("/chat")` (line ~809)

- Validates the conversation exists and belongs to the user.
- Returns `StreamingResponse(service.stream_turn(...), media_type="text/event-stream")` — that's the SSE channel.

### Step B — per-turn pipeline
**File:** same → `ChatService.stream_turn()` (line ~653)

1. Loads / creates the in-process **Tier-1 buffer** for the conversation (`ShortTermBuffer`).
2. **Hydrates** the buffer from SQLite if cold (`_hydrate`).
3. Persists the user message to **Tier-2 SQLite** (`SQLiteStore.append_message`).
4. Asks **Tier-3 Mem0** for relevant prior facts (`memory.search`).
5. Yields `event: start`.
6. Streams the agent's deltas as `event: chunk`s.
7. Persists the assistant message to SQLite.
8. **Asynchronously** writes the (user, assistant) pair to Mem0 — fire-and-forget so it stays off the critical path.
9. Yields `event: done`.

### Step C — orchestrator agent
**File:** same → `OrchestratorAgent.astream()` (line ~498)

This is the heart of the system. Four steps:

1. **Safety guard** — `await self.safety.check(text)`.
   - Loads the ONNX model lazily on first call (`FinancialSafety._ensure_loaded`).
   - Runs `safety.inference.FinancialGuardrail.classify`.
   - If `should_block`, yields a single `{blocked, category, reason}` JSON chunk and **returns** — classifier never runs.

2. **User context lookup** — `self.users_db.get(user_id)`.
   - Reads `portfolio_check/users.json` once and caches.
   - Production swap point: replace `UsersDB._load` with a Postgres / Mongo call.

3. **Intent classifier** — `intent_classifier(user_ctx_str, text)`.
   - File: `src/intent_classifier.py`.
   - One `gpt-4o-mini` (dev) / `gpt-4.1` (eval) call, `temperature=0`, `max_tokens=150`.
   - Returns structured JSON: `{agent, entities}`.
   - On any exception → `_tfidf_fallback(query)` → cosine similarity over intent definitions.

4. **Dispatch** based on `agent_name`:
   - `"portfolio_health"` → `_run_portfolio_health(user_context)`.
   - `"general_query"`    → `_run_general_query(messages, memories)` — streams `gpt-4o-mini`.
   - Anything else in `ALL_INTENTS` → yields `{"status": "agent_not_built", ...}`.

### Step D — Safety guard internals
**File:** `src/safety/inference.py`

- `FinancialGuardrail.__init__` loads the INT8-quantized ONNX model + tokenizer from `bert_onnx/`.
- `classify(query)` runs the model, picks the argmax label, and routes:
  - label ends with `_op`  → `should_block=True`
  - label ends with `_edu` → `should_block=False`
  - label is `general`     → `should_block=False`, picks the closest `_edu` topic for `category`.
- `compose_reason(label, conf)` turns the verdict into a natural-language explanation (operational vs. educational, with confidence).

### Step E — Portfolio Health agent
**File:** `src/portfolio_check/portfolio_check/agent.py` → `run_portfolio_check(user_context)`

1. `_validate_user_context` — must have `user_id`.
2. **Empty portfolio short-circuit** → `_build_empty_portfolio_response` (BUILD path, no MCP).
3. Otherwise open `_data_session()`:
   - default: `mcp_client.yfinance_session()` — spawns `uvx yfmcp@latest` over stdio.
   - if `PORTFOLIO_DATA_BACKEND=local`: in-process `yfinance` (no MCP server needed).
4. `compute_all(...)` in `metrics.py` — *all numbers come from here*, nowhere else.
5. `_generate_observations(...)` — passes the *computed* numbers to `gpt-4o-mini` and asks for 1–3 plain-language observations. The LLM **never produces numbers**.
6. `_format_response(...)` — assembles the final JSON, attaches `DISCLAIMER`.

### Step F — Memory tiers
**File:** `chat_agent.py`

- `ShortTermBuffer` — `OrderedDict[conversation_id, deque[Message]]`, LRU-evicted. Sub-ms reads.
- `SQLiteStore` — `aiosqlite`, WAL mode. Schema in the `SCHEMA` constant (users, conversations, messages with FKs).
- `Mem0Store` — async wrapper around `mem0.Memory`, pinned to `gpt-4o-mini` + `text-embedding-3-small`. Search and add are both `asyncio.to_thread`'d so the event loop never blocks.
- `_build_memory(enabled)` — try Mem0; fall back to `NoopMemoryStore` on any failure.

---

## 8. Reproducing the safety model from scratch (optional)

The trained ONNX artefact ships in the repo. If you want to retrain:

```bash
cd src/safety

# 1. Build the dataset (uses raw/ — beaver_*.jsonl.gz, agentharm.json)
python build_dataset_bert.py
#   → data_bert/{train,val,test}.jsonl + labels.json

# 2. Fine-tune BERT-mini (8 epochs, early stopping on harmful_recall)
python train_bert.py
#   → bert_ckpt/

# 3. Export to ONNX + INT8 quantize
python export_onnx.py
#   → bert_onnx/model_quantized.onnx

# (or do steps 2+3 in one shot)
python train_and_export.py

# 4. Sanity-check on the test set
python eval_bert.py
```

On an Apple silicon laptop this takes ~5–10 min end-to-end. On CPU-only Linux, ~30 min.

---

## 9. Common gotchas

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: mem0` | Optional dep failed to install. Service still works (Noop fallback). To enable: `pip install mem0ai`. |
| `uvx: command not found` when classifier routes to portfolio_health | Install `uv` (`pip install uv`) — it provides `uvx`. Or set `PORTFOLIO_DATA_BACKEND=local` to use in-process `yfinance`. |
| Safety guard slow on first request | One-time ONNX model load (~1–2s). Subsequent requests are ~4ms. The lazy load is intentional so app boot stays fast. |
| `chat.db` keeps growing | SQLite is the durable Tier-2 store — it's expected. Wipe with `rm src/chat_agent/chat.db` between local demos. |
| OpenAI rate-limit errors | Drop `MODEL_DEV` to `gpt-4o-mini` (default in dev) and check Mem0 isn't pinned to a higher-tier model. |

---

## 10. Where to extend

- **New specialist agent.** Add a branch to `OrchestratorAgent.astream` and add the intent name to `WIRED_INTENTS`. The classifier already returns the label; the router already knows to fall through to `agent_not_built` for unwired intents — so the only thing missing is your handler.
- **New safety category.** Edit `src/safety/build_dataset_bert.py` to emit the new label, retrain, re-export. The runtime (`inference.py`) auto-discovers labels via `labels.json`.
- **Production users DB.** Replace `UsersDB._load` with a Postgres / Mongo query. Everything downstream is unchanged.
