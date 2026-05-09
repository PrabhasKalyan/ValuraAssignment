[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SHM9MYZJ)

# Valura AI — AI Microservice

An AI microservice for a global wealth management platform: a single chatbot whose internals are a small ecosystem of specialist agents. A query enters the HTTP layer, gets filtered for safety, classified into an intent, routed to the right agent, and streamed back to the client over SSE.

The hard problem isn't "build a chatbot." The hard problem is building one for **financial services** — where the chatbot must answer educational queries freely (*"what is insider trading?"*) while refusing the operationally identical, harmful version (*"how do I do insider trading?"*). The whole architecture below is shaped by that single distinction, and by the latency / cost ceilings that come with running it for real users.

---

## 1. Architecture at a glance

```
                       ┌─────────────────────────────────────┐
   POST /chat  ───────▶│  HTTP layer (FastAPI, SSE stream)   │
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  1. Safety Guard (BERT-mini, ONNX)  │   ~4ms, no LLM, no network
                       │     {blocked, category, reason}     │
                       └──────────────────┬──────────────────┘
                                          │ (if not blocked)
                       ┌──────────────────▼──────────────────┐
                       │  2. User-context lookup             │   users.json (prod: Postgres/Mongo)
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  3. Intent Classifier (1 LLM call)  │   structured JSON
                       │     fallback: TF-IDF cosine sim     │   no LLM, no regex
                       └──────────────────┬──────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
   ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
   │ portfolio_health     │  │ general_query        │  │ <other intents>      │
   │ (yfinance MCP)       │  │ (gpt-4o-mini stream) │  │ stub: agent_not_built│
   └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
              └───────────────────────────┼───────────────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  SSE stream  (start / chunk / done) │
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  3-tier memory:                     │
                       │  buffer (RAM) + SQLite + Mem0       │
                       └─────────────────────────────────────┘
```

Files that own each box:

| Component                    | Path |
|---|---|
| HTTP layer + orchestrator    | `src/chat_agent/chat_agent.py` |
| Safety guard (BERT-mini)     | `src/safety/inference.py` (+ training scripts in `src/safety/`) |
| Intent classifier            | `src/intent_classifier.py` |
| Portfolio Health agent       | `src/portfolio_check/portfolio_check/agent.py` |
| yfinance MCP client          | `src/portfolio_check/portfolio_check/mcp_client.py` |
| Deterministic metrics layer  | `src/portfolio_check/portfolio_check/metrics.py` |
| User context (sample DB)     | `src/portfolio_check/portfolio_check/users.json` |

---

## 2. Safety Guard — *why a tiny transformer, not a regex*

The assignment's hard requirement: **no LLM call, no network call, < 10ms per input.** And on top of that, the guard must distinguish:

> *"What is insider trading?"*  → **allow** (educational)
> *"How do I do insider trading?"*  → **block** (operational)

Those two sentences share almost every keyword. Walking through the option space:

| Approach | Verdict |
|---|---|
| Big LLM (GPT-4o-mini, Claude, etc.) | **Rejected.** Network call. Latency budget is 10ms. |
| Local LLM (≥1B–3B params) | **Rejected.** Even a 1B model is hundreds of ms on CPU and is overkill for a binary-ish decision. |
| Rule-based regex / keyword matching | **Rejected.** Cannot tell *"what is X"* from *"how do I X"* once you account for paraphrase. The keyword list is unbounded — every variant the attacker tries is a new entry. |
| Classical ML (logistic regression on bag-of-words / TF-IDF) | **Rejected.** It can detect *topic*, but not the **operational vs. educational** axis. Both queries have the same word distribution. |
| **Tiny transformer with multi-head self-attention** | **Adopted.** Self-attention reads the *whole sentence* and learns the dependency between *"how do I"* and the harmful noun phrase. That is exactly the signal the linear/keyword approaches cannot model. |

### What we actually shipped

- **Backbone:** `prajjwal1/bert-mini` — 11M params, 4 layers, 256 hidden. Tiny by transformer standards, large enough to encode whole-sentence context.
- **Head:** a 13-class classification head whose labels factor topic × intent: `{insider_trading, market_manipulation, money_laundering, guaranteed_returns, reckless_advice, sanctions_evasion} × {op, edu}` plus a `general` class. The classifier surfaces three things at once — a **boolean block decision**, the **category**, and a **reason** — by reading those labels.
- **Export:** ONNX + dynamic INT8 quantization → `src/safety/bert_onnx/model_quantized.onnx`. Runs on CPU.
- **Measured latency:** ~**4ms / inference** on an M-series CPU (well under the 10ms budget).
- **Output shape:**
  ```json
  {
    "should_block": true,
    "category": "insider_trading",
    "reason": "Blocked: this query reads as an OPERATIONAL request — i.e. the user appears to be asking how to perform trading on non-public material information. Confidence: 0.91."
  }
  ```

### Why we had to build the dataset

BERT-mini is open source. Public *financial-guardrail* datasets are not. We assembled a **semi-synthetic** training set by:

1. Pulling structurally similar samples from public Hugging Face datasets (PKU-Beaver, AgentHarm — see `src/safety/raw/`).
2. Running a Python pipeline (`src/safety/build_dataset_bert.py`) that filters, re-templates, and balances those into the 13-class schema we actually need.
3. Producing `train.jsonl / val.jsonl / test.jsonl` under `src/safety/data_bert/`.

Then `src/safety/train_bert.py` fine-tunes BERT-mini for 8 epochs with early stopping on `harmful_recall`. `src/safety/export_onnx.py` runs the ONNX export + INT8 quantization. The trained artefacts live in `src/safety/bert_ckpt/` and `src/safety/bert_onnx/`.

### Trade-off we accepted

The guard is allowed to **over-block** ambiguous edges. If a query has the structural signature of an operational request and confidence is below the block threshold, we **fail safe** (still block, with a low-confidence reason string). The cost of one false block is a polite refusal; the cost of one false allow is a regulatory incident. The asymmetry is the point.

---

## 3. Intent Classifier — *one LLM call, with a fallback that is not an LLM*

The classifier must do four things in **one call**:

1. Pick exactly one of 10 intents (`portfolio_health`, `market_research`, `investment_strategy`, `financial_planning`, `financial_calculator`, `risk_assessment`, `product_recommendation`, `predictive_analysis`, `customer_support`, `general_query`).
2. Extract entities (tickers, amounts, currency, rate, period_years, frequency, horizon, time_period, topics, sectors, index, action, goal).
3. Decide which agent to dispatch.
4. Return a structured safety verdict (informational only — the guard is the only authority that blocks).

### Personalization via user context

Routing isn't pure NLP — *"how am I doing?"* means **portfolio_health** for a user who has positions, and **general_query / financial_planning** for a user with an empty account. So the classifier needs the user's profile.

- The HTTP layer takes a `user_id`.
- `UsersDB` (`chat_agent.py`) loads `src/portfolio_check/portfolio_check/users.json` and looks up the matching profile.
- The profile is serialized to JSON and passed into the prompt alongside the query.

> **Production note.** `users.json` is the demo persistence. In production, this is a Postgres/Mongo lookup behind the same `UsersDB` interface — only the `_load` method changes. The rest of the code does not care.

### Prompting strategy

A single call to `gpt-4o-mini` (dev) / `gpt-4.1` (eval) with:
- `temperature=0`, `max_tokens=150`
- A system prompt that defines all 10 intents, lists every entity field with its normalization rule, and **mandates JSON-only output**
- A user prompt of `query: ...` + `user_context: ...`

The full prompt is in `src/intent_classifier.py`.

### Failure mode — what happens when the LLM call fails

The assignment requires **no crash** when the LLM is unavailable. Options considered:

| Fallback | Verdict |
|---|---|
| Another transformer | **Rejected.** Same latency / weight cost we just avoided in the safety layer; redundant. |
| Rule-based regex on intent keywords | **Rejected.** Users do not phrase queries to match keywords. Maintaining the keyword list is unbounded work. |
| **TF-IDF + cosine similarity over intent definitions** | **Adopted.** Tiny memory footprint, no model weights, deterministic. Good enough as a *floor* — the LLM is the real classifier; this just keeps the request alive. |

The fallback fits in ~10 lines (`_tfidf_fallback` in `intent_classifier.py`). It vectorizes the query with the same `TfidfVectorizer` already fit on the intent definitions and returns the argmax. If similarity to every intent is below `0.05`, it falls through to `general_query` — a safe default that always has a real agent behind it.

---

## 4. Portfolio Health Check Agent

The first specialist agent and the one a novice hits first when they want to know "is everything OK?". Speaks to **MONITOR** and **PROTECT**.

### Pipeline (`src/portfolio_check/portfolio_check/agent.py`)

1. Validate user context. **Empty portfolio → BUILD path** (no MCP calls; the LLM produces 1–3 starter considerations tailored to age / country / risk profile / base currency).
2. Open a **yfinance MCP** session (`mcp_client.py`, stdio transport, `uvx yfmcp@latest` by default).
3. Fetch live quotes, FX rates, and benchmark history.
4. **Compute every metric deterministically** in `metrics.py` (Python, no LLM): per-position weight, return, HHI concentration, top-1 / top-3 share, total / annualized return, alpha vs. benchmark.
5. Pass the *computed* numbers to the LLM and let it write **only prose** — 1–3 plain-language observations, with severity (`warning` / `info` / `positive`), tailored to the user's risk profile.
6. Attach a regulatory disclaimer.

The LLM **never produces numbers** — that's the load-bearing rule. Hallucinated returns are out of the question by construction.

### Output shape

```json
{
  "user_id": "user_003_concentrated",
  "status": "ok",
  "base_currency": "USD",
  "as_of": "2026-05-05",
  "holdings": [...],
  "concentration_risk": {
    "top_position_pct": 60.4,
    "top_position_ticker": "NVDA",
    "top_3_positions_pct": 78.2,
    "hhi": 0.41,
    "flag": "high"
  },
  "performance": {
    "total_value": 215430.0,
    "total_cost": 182000.0,
    "total_return_pct": 18.4,
    "annualized_return_pct": 12.1,
    "weighted_holding_days": 540
  },
  "benchmark_comparison": {
    "benchmark": "S&P 500",
    "portfolio_return_pct": 18.4,
    "benchmark_return_pct": 14.2,
    "alpha_pct": 4.2,
    "period_start": "2024-01-15",
    "period_end": "2026-05-05"
  },
  "observations": [
    {"severity": "warning", "text": "60% of your portfolio is in NVDA — that's a heavy concentration for one company."},
    {"severity": "info",    "text": "You're outperforming the S&P 500 by 4.2% over the period."}
  ],
  "disclaimer": "This is not investment advice. ..."
}
```

The four contractual fields — **concentration risk**, **benchmark comparison relevant to the user's market**, **performance metrics**, and **specific actionable observations grounded in the user's actual holdings** — are all present and load-bearing. Observations are capped at three and ordered by importance so a novice reads the *one or two things that matter most*, not a metric dump.

### Empty portfolio (`user_004_empty`)

Skips MCP entirely. Returns a `status: "ready_to_build"` response with starter-oriented observations (broad index funds, bond allocation, emergency fund — no specific tickers). Does not crash.

---

## 5. Session Memory — *three tiers, dynamic by lifetime*

Spec lets us pick any persistence; we picked a hybrid because no single tier covers all the things memory has to do.

| Tier | Backed by | Lifetime | Purpose |
|---|---|---|---|
| **1. Short-term buffer** | in-process `deque` (`ShortTermBuffer`) | session / hot | last N turns, sub-ms reads, feeds the prompt directly |
| **2. Durable raw history** | SQLite (`aiosqlite`, WAL) | forever | every message, write-through; survives restarts; hydrates the buffer on cold conversations |
| **3. Long-term semantic facts** | **Mem0** (open-source library, OpenAI-backed) | forever | extracts the *important* facts from history and answers semantic queries about them |

### Why all three

- **Buffer alone:** loses everything on restart, blows past memory under load.
- **SQLite alone:** you have to load *every prior turn* to keep context. Cost climbs linearly per turn, then so does latency, then so do hallucinations as the prompt drifts.
- **Mem0 alone:** great for facts, terrible as a turn-by-turn buffer (latency, indirection).
- **All three together:** the buffer handles short conversations cheaply; SQLite is the source of truth; Mem0 cuts through long histories by surfacing only what's relevant to *this* query.

### Why Mem0 specifically

Two specific behaviours make Mem0 the right Tier-3:

1. **Topic-switch handling.** When the user pivots from topic A to topic B, naively replaying the last N turns drags A back into the prompt. Mem0's semantic search returns prior turns **scored against the current query**, so it pulls B-relevant context and ignores A.
2. **Ambiguous follow-ups.** *"What about Apple?"* after *"tell me about Microsoft"* needs the prior turn; *"what about Apple?"* in isolation does not. Mem0's relevance score lets us inject prior context only when it's actually relevant, instead of blindly stuffing the prompt.

We use the **open-source Mem0 library** integrated with the OpenAI API, **not the hosted Mem0 service** — no extra vendor in the trust boundary, no extra credentials. Pinned to `gpt-4o-mini` + `text-embedding-3-small` so it costs cents and so we don't hit the `max_tokens` vs. `max_completion_tokens` issue that some newer OpenAI reasoning models surface.

If `mem0ai` is not installed or fails to initialize, the system degrades cleanly to a `NoopMemoryStore` (`Mem0Store` → `NoopMemoryStore` fallback in `_build_memory()`). Tiers 1 and 2 still work.

---

## 6. HTTP Layer & Streaming

FastAPI service in `src/chat_agent/chat_agent.py`. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/users` | upsert a user |
| `POST` | `/conversations` | create a conversation |
| `GET`  | `/conversations?user_id=…` | list user's conversations |
| `GET`  | `/conversations/{id}/messages` | replay a conversation |
| `POST` | `/chat` | **the main endpoint — SSE stream** |

`/chat` is the only response path. It runs `safety → user-context lookup → classifier → routed agent` and streams the result back as SSE. Three event types:

- `event: start` — `{message_id, memories_used}`
- `event: chunk` — `{text: "..."}` — repeated, one per delta
- `event: done`  — `{message_id, content, buffer_size}`
- `event: error` — `{message: "..."}` — on failure, never a stack trace

The streaming is real streaming — `OpenAI.chat.completions.create(..., stream=True)` deltas are forwarded directly into the SSE chunks. Nothing is buffered into a single JSON response. Headers are set to defeat proxy buffering (`Cache-Control: no-cache`, `X-Accel-Buffering: no`).

### Stub agents

`OrchestratorAgent.WIRED_INTENTS = {"portfolio_health", "general_query"}`. Every other intent in `ALL_INTENTS` returns a clean structured response:

```json
{
  "status": "agent_not_built",
  "intent": "market_research",
  "entities": {...},
  "message": "No agent is wired for intent 'market_research' yet."
}
```

The router does not crash on unimplemented agents. Adding a new agent later is two lines: import it, add a branch.

### Safety precedence

The guard runs **first**. If it blocks, the classifier never runs and a single `{blocked, category, reason}` chunk goes out. The classifier may *also* return a safety verdict in its structured output — that field is informational only, never re-blocks, never re-routes.

---

## 7. Setup

**Requirements:** Python 3.11+. An `OPENAI_API_KEY` for `/chat` against real LLMs.

```bash
git clone <repo>
cd valura-ai-ai-engineer-assignment-PrabhasKalyan-main

python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt

cp .env.example .env
# Fill in OPENAI_API_KEY
```

Run the service:

```bash
uvicorn src.chat_agent.chat_agent:app --reload
```

### Environment variables

| Var | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for the classifier, general-query agent, portfolio observations, and Mem0 |
| `PORTFOLIO_DATA_BACKEND` | `mcp` | `mcp` = stdio MCP server (default), `local` = in-process yfinance |
| `PORTFOLIO_MCP_CMD` | `uvx` | command to spawn the MCP server |
| `PORTFOLIO_MCP_ARGS` | `yfmcp@latest` | args for the MCP server |
| `PORTFOLIO_LLM_MODEL` | `gpt-4o-mini` | LLM used by the portfolio agent for prose only |

`gpt-4o-mini` for development; switch to `gpt-4.1` for evaluation.

---

## 8. Running tests

```bash
pytest tests/ -v
```

Tests run **without** an `OPENAI_API_KEY` — the LLM is mocked. Test files:

- `tests/test_safety_pairs.py` — runs the guard against `fixtures/test_queries/safety_pairs.json` and verifies recall on harmful queries / pass-through on educational ones.
- `tests/test_classifier_routing.py` — runs the classifier against `fixtures/test_queries/intent_classification.json` with subset entity matching.
- `tests/test_portfolio_health_skeleton.py` — exercises the agent on every fixture user, including `user_004_empty` (must not crash).

---

## 9. Cost & performance — how we measured

| Target | Result |
|---|---|
| p95 streaming first-token latency | < 2s (safety ~4ms + classifier 1 LLM call → first delta from the routed agent) |
| p95 end-to-end response time | < 6s |
| Cost per query (`gpt-4.1` pricing) | < $0.05 |

How:
- One LLM call for classification, one stream for the answer (or zero LLM calls when the safety guard blocks). No hidden second classifier, no hidden reranker.
- Safety guard is INT8-quantized ONNX on CPU — measured at ~4ms / inference.
- Mem0 writes happen in `asyncio.create_task(...)` *after* the SSE `done` event — they are off the critical path.
- `max_tokens=150` on the classifier; the answer streams so the user sees the first token long before the last.

---

## 10. What is not in scope

- Only **portfolio_health** and **general_query** are wired. Every other intent in the taxonomy returns a clean stub. Adding one is a one-branch extension in `OrchestratorAgent.astream`.
- `users.json` is a flat file. Production swaps in Postgres / Mongo behind the same `UsersDB` interface.
- The Mem0 fallback to `NoopMemoryStore` means the service still works without the optional dep — Tiers 1 + 2 carry the load.

---

## 11. Defence video

`<video link to be added on final commit>`
