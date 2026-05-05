# Fixtures

Sample user-side data for the Valura AI assignment.

You will not find market data, prices, sector classifications, or benchmarks here. Get those from MCP servers, the `yfinance` package, or any source you choose. **Do not hardcode market data into your code.**

The data is global — US, UK, EU, Japan, Singapore. Tickers use proper exchange suffixes (`AAPL`, `ASML.AS`, `HSBA.L`, `7203.T`) so they resolve against any market data provider.

---

## Layout

| Directory | Purpose |
|---|---|
| `users/` | 5 user profiles with portfolios, each chosen to surface a different edge case |
| `conversations/` | 3 multi-turn test cases. Use these to test follow-up resolution and topic switching |
| `test_queries/` | Labeled query sets — the gold standard for classifier and safety-guard testing |

---

## Users

| File | Edge case |
|---|---|
| `user_001_active_trader_us.json` | Aggressive US trader, 9 holdings, tech-heavy |
| `user_003_concentrated.json` | ~60% of portfolio in a single stock — concentration risk |
| `user_004_empty.json` | KYC complete, zero positions — agent must not crash |
| `user_006_multi_currency.json` | Singapore-based, USD + EUR + GBP + JPY holdings — multi-currency normalization |
| `user_008_retiree.json` | Dividend-focused retiree, conservative — agent should weight commentary toward yield |

All files use the same shape. If you write a Pydantic model, derive it from one example and validate against the others.

---

## Conversations

Each file contains a `test_cases[]` array. Every test case provides:
- `prior_user_turns[]` — the conversation history (user turns only) leading up to the current turn
- `current_user_turn` — the query your classifier should classify
- `expected.agent` and `expected.entities` — the gold-standard routing

| File | What it tests |
|---|---|
| `follow_up_session.json` | Pronoun and entity carryover ("how much do I own?" after "tell me about NVDA") |
| `multi_intent_session.json` | Topic switches — context must NOT carry inappropriately |
| `ambiguous_session.json` | Typos, vague references, missing parameters |

---

## Test queries

| File | Format |
|---|---|
| `intent_classification.json` | `{query, expected_agent, expected_entities}` for ~60 queries |
| `safety_pairs.json` | `{query, should_block, category}` for ~45 queries (mixed harmful + educational) |

---

## Matching rules (for grading)

Your classifier output is matched against the gold files using the following rules:

**Agent (`expected_agent`):** exact string match against the taxonomy in `intent_classification.json`.

**Entities (`expected_entities`):** subset match with normalization. Your output must contain every value listed; extra values are allowed.

| Field | Normalization rule |
|---|---|
| `tickers` (array) | Case-folded; exchange-suffix optional (`AAPL` matches `aapl` and `AAPL.US`) |
| `topics` / `sectors` (arrays) | Case-folded; exact substring match per element |
| `amount` (number) | Within ±5% |
| `rate` (number) | Within ±5% |
| `period_years` (number) | Exact |
| `currency` (string) | ISO 4217, exact |
| `index` (string) | Exact match against the canonical name (`S&P 500`, `FTSE 100`, `NIKKEI 225`, `MSCI World`) |
| `action`, `goal`, `frequency`, `horizon`, `time_period` | Exact match against the vocabulary in `entity_vocabulary` |

These rules are open. Implement them in your `tests/` matcher. We use the same rules during evaluation.

---

## Open vs hidden test sets

These fixtures are **open**. We will run a **separate, larger labeled set** during evaluation. Optimizing only against the public set will hurt your score — the hidden set covers the same vocabulary and rules but with novel queries.
