# Valura AI — Team Lead Project Assignment

---

## Context

Valura is a global wealth management platform. The AI microservice is the intelligence layer behind every AI interaction on the platform.

**The mission of this microservice is to be the AI co-investor for every user — especially novices.** It should help any investor, regardless of experience, do four things over the long arc of their financial life:

| | What it means in practice |
|---|---|
| **BUILD** | Help a new investor go from zero to a first allocation: understand goals, pick instruments, size positions |
| **MONITOR** | Tell the user what's actually happening in their portfolio in plain language — performance, drift, risk, news |
| **GROW** | Suggest specific, grounded next moves — rebalances, additions, opportunities aligned to their risk profile |
| **PROTECT** | Surface concentration, drawdown, leverage, and behavioural risks before they hurt — and refuse anything reckless |

Each user interaction is handled by a small ecosystem of specialist agents (market research, planning, calculations, risk, recommendations, predictive analysis, portfolio health, support). One classifier decides which agent runs. The whole pipeline streams back in real time.

You are applying to **lead the team that owns this microservice end to end** — architecture, implementation, reliability, and direction.

This assignment is a slice of that real system. We're not asking you to build the whole ecosystem — we're asking you to build the **spine** (safety + classifier + routing + one fully-implemented agent + the HTTP layer) such that adding more agents later is a straight extension, not a rewrite.

Build it the way you would build it on day one of the job.

---

## Rules

- **3 days** from receipt
- **Python 3.11+.** Any libraries you choose — justify your choices in your README
- **Self-host everything you need.** We do not provide credentials
- **Streaming response is required** (Server-Sent Events)
- **Single `README.md`** at repo root — all your reasoning, decisions, and instructions go here. No separate design docs
- **Incremental commits required.** We read the git log. A handful of large commits is acceptable; one final dump is not
- **Submission video required** (see *Defence* below)
- All tests must pass with `pytest tests/ -v`. We will run them
- Tests must run in CI **without** an `OPENAI_API_KEY` — mock the LLM

---

## Cost & Performance Targets

These are evaluated, not just stated. A submission that ignores them will be marked down.

| Target | Value |
|---|---|
| Model used during development | `gpt-4o-mini` (lower cost) |
| Model used during evaluation  | `gpt-4.1` |
| p95 streaming first-token latency | < 2s |
| p95 end-to-end response time | < 6s |
| Cost per query (at `gpt-4.1` pricing) | < $0.05 |

Document how you measured these in your README.

---

## The System

A FastAPI microservice that receives user queries — financial questions, portfolio requests, market research — classifies the intent, routes the query to the right specialist agent, and streams the response back to the user.

**User context** — portfolio holdings, KYC status, risk profile — is passed into the pipeline.

**Session memory:** agents can see prior turns of the same conversation. Persistence is **your choice** — Postgres, SQLite, or in-memory for the demo. Justify your pick in the README. We will not penalize an in-memory implementation if you defend the tradeoff.

**Responses stream** to the client via SSE. Pick an SSE library (e.g. `sse-starlette`) or implement the protocol yourself — your call.

**Safety** is non-negotiable: refuse insider trading, market manipulation, money laundering, guaranteed-return claims, reckless advice. Do not refuse educational questions about those topics.

**The intent classifier** is a single LLM call that returns the user's intent, all extracted entities (tickers, amounts, time periods), which agent to dispatch, and an informational safety verdict — all in one structured output.

---

## Provided Data

The repo ships with sample user-side data in `fixtures/`:

- **5 user profiles** covering edge cases: aggressive trader, concentrated single-stock holder, empty portfolio, multi-currency global investor, dividend-focused retiree
- **3 conversation transcripts** (in test-case format) for testing follow-up resolution and topic-switch handling
- **Labeled query sets** — ~60 classification queries and ~45 safety queries — that serve as your gold standard for testing

Read `fixtures/README.md` first. **Do not hardcode market data** (prices, sectors, fundamentals, benchmarks) into your code — fetch it from MCP servers, the `yfinance` package, or any source you choose.

**MCP fluency is a plus, not required.**

---

## What to Build

Three components plus the HTTP layer that ties them together.

### 1. Safety Guard

The synchronous filter that runs before the LLM is called.

- No LLM call. No network call. Pure local computation
- Must complete in well under 10ms for any input
- Blocks obvious harmful intent across the categories in `fixtures/test_queries/safety_pairs.json`
- Each blocked category returns a distinct, professional response — not a generic refusal
- Edge cases (e.g. educational queries on harmful topics) may be over-blocked; document your tradeoff in the README

### 2. Intent Classifier

The single LLM call that drives the entire pipeline.

- One LLM call per classification
- Returns a structured output (your choice of schema) covering: intent, extracted entities, target agent, informational safety verdict
- An LLM failure must not crash the request — define the fallback behaviour
- Must handle follow-up queries that reference the previous turn ("what about Apple?" after "tell me about Microsoft")
- Must handle the conversation test cases in `fixtures/conversations/`

### 3. Portfolio Health Check Agent

The first specialist agent — and the one a novice investor will hit first when they want to know "is everything OK?". It speaks to the **MONITOR** and **PROTECT** halves of the mission.

When a user asks "how is my portfolio doing?", "give me a health check", "am I diversified?", or similar, this agent runs.

- The agent receives the user's portfolio data as input. It does not fetch it itself
- It produces a **structured output** covering at minimum: concentration risk, benchmark comparison relevant to the user's market, performance metrics, and specific actionable observations grounded in the user's actual holdings
- Observations should be useful to a novice — plain language, no jargon without context, surface the *one or two things that matter most* rather than dumping every metric
- It handles users with no portfolio without crashing — for `user_004_empty`, the agent should produce a useful response oriented toward **BUILD** (this user is ready to start; what should they consider?), not an error
- Every response includes a regulatory disclaimer
- Queries about portfolio health are routed here by the classifier

**Reference output shape** (you may extend or rename fields, but the structure should be at least this rich):

```json
{
  "concentration_risk": {
    "top_position_pct": 60.4,
    "top_3_positions_pct": 78.2,
    "flag": "high"
  },
  "performance": {
    "total_return_pct": 18.4,
    "annualized_return_pct": 12.1
  },
  "benchmark_comparison": {
    "benchmark": "S&P 500",
    "portfolio_return_pct": 18.4,
    "benchmark_return_pct": 14.2,
    "alpha_pct": 4.2
  },
  "observations": [
    {"severity": "warning", "text": "60% of portfolio in NVDA — highly concentrated."},
    {"severity": "info",    "text": "Outperforming S&P 500 by 4.2% over the period."}
  ],
  "disclaimer": "This is not investment advice. ..."
}
```

### 4. HTTP Layer

The FastAPI application that exposes the system.

- One endpoint that accepts a user query and runs the full pipeline: safety guard → classifier → routed agent → streamed response
- **Streaming via SSE is the only response mode.** No JSON fallback path
- Errors return structured SSE error events, not stack traces
- The pipeline enforces a sane timeout — pick a number and defend it

---

## Stub Contract for Unimplemented Agents

You implement Portfolio Health end-to-end. For all **other** agents named in `fixtures/test_queries/intent_classification.json` (market_research, investment_strategy, financial_calculator, etc.), the router must still work.

For these, return a structured "not implemented" response that includes:
- The classified intent
- The extracted entities
- The agent that would have handled this
- A short message indicating the agent is not implemented in this build

Do not crash. Do not return errors. The router's job is to route correctly even when the destination is a stub.

---

## Safety Precedence

The safety guard runs **first**. If it blocks, the classifier never runs.

If the guard passes, the classifier may also return a safety verdict in its structured output. This verdict is **informational only** — it appears in the response metadata but does not change routing or trigger a re-block. The guard is the only authority that blocks a query.

---

## Testing Contract

Your tests must work against the provided gold files in `fixtures/test_queries/`.

**Routing match:** Your classifier's chosen agent must equal `expected_agent` exactly (string match against the taxonomy in `intent_classification.json`).

**Entity match:** Tested as **subset match with normalization**.
- For string lists (tickers, topics, sectors): your output must contain every value listed in `expected_entities`. Extra values are allowed
- Normalization rules apply per field — see `fixtures/README.md`. Tickers are case-folded and exchange-suffix is optional (`AAPL` matches `aapl`; `ASML` matches `ASML.AS`)
- Numeric fields (amount, rate, period_years) match within ±5%
- Document your matcher in `tests/`

**Success thresholds** (graded):
| Metric | Threshold |
|---|---|
| Classifier routing accuracy | ≥ 85% |
| Safety guard recall on harmful queries | ≥ 95% |
| Safety guard pass-through on educational queries | ≥ 90% |
| Portfolio Health response on `user_004_empty` | must not crash, must include a sensible message |

We will run a **separate, larger labeled set** during evaluation. Optimizing only against the public set will hurt your score.

---

## Optional Stretch (not required, not graded as failures)

If you finish early, demonstrate one or more:
- Identical-query LLM dedupe cache (intra-session)
- Embedding-based pre-classifier (skip the LLM call when confidence is high)
- Per-tenant model selection (e.g. premium users → `gpt-4.1`, free → `gpt-4o-mini`)
- Multi-tenant rate limiting

---

## Defence

Within 24 hours of pushing your final commit, upload an **unlisted YouTube video** (or equivalent) walking us through your submission.

**Hard rules:**
- **Maximum 10 minutes.** Submissions over 10 minutes are auto-rejected
- Cover: architecture (how a request flows), one non-obvious decision and why, one thing you'd do differently with another week
- Link the video URL in your `README.md`

We watch every video before reviewing the code.

---

## Hard Constraints

- All code in `src/`
- All tests in `tests/`. Use pytest. Mock the LLM in tests — CI must run without `OPENAI_API_KEY`
- No secrets in the repo. Use `.env` (gitignored); document required variables in `.env.example`
- No copy-pasted scaffolds you can't explain in the video

---

## Deliverables

```
README.md          — single source of truth: setup, env vars, library choices, decisions, video link
src/               — all code
tests/             — all tests, must pass with pytest
```

---

## What we are looking for

Someone who reads a system description and immediately sees the failure modes — before a line of code is written.

Someone who builds things that hold up at the edges, not just the happy path.

Someone who keeps the **end user** — a novice investor trying to build, monitor, grow, and protect their wealth — in mind while making technical tradeoffs. The right architecture for that user is not always the most elegant one.

Someone whose code review feedback would actually make the codebase better, because they know the difference between a stylistic preference and a real risk.

Someone whose README and 10-minute video both make us think: "this person can run the team."
