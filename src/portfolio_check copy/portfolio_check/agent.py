"""
Portfolio health agent.

Pipeline:
  1. Validate user_context. Empty portfolio → BUILD response (no MCP calls).
  2. Connect to yfinance MCP server.
  3. Fetch live quotes + FX + benchmark history.
  4. Compute metrics deterministically (see metrics.py).
  5. Ask the LLM to turn the numbers into 1-3 plain-language observations
     tailored to the user (age, risk profile, country, income focus).
  6. Attach disclaimer and return.

The LLM only writes prose — it never produces numbers. Every metric the user
sees comes from the deterministic layer, so the response can't hallucinate
returns or weights.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import date
from typing import Any

from openai import AsyncOpenAI

from .metrics import PortfolioMetrics, compute_all


def _data_session():
    """Choose the yfinance backend.

    PORTFOLIO_DATA_BACKEND=local → in-process yfinance (no MCP server needed)
    PORTFOLIO_DATA_BACKEND=mcp   → stdio MCP server (default, production path)
    """
    backend = os.getenv("PORTFOLIO_DATA_BACKEND", "mcp").lower()
    if backend == "local":
        from .local_yfinance import local_session
        return local_session()
    from .mcp_client import yfinance_session
    return yfinance_session()


DISCLAIMER = (
    "This is not investment advice. The figures above are computed from "
    "publicly available market data and the holdings you provided; they "
    "may be delayed or incomplete. Past performance does not guarantee "
    "future results. Consult a licensed financial advisor before acting "
    "on any of this information."
)

OBSERVATIONS_MODEL = os.getenv("PORTFOLIO_LLM_MODEL", "gpt-4o-mini")


# --- public entrypoint -----------------------------------------------------


async def run_portfolio_check(user_context: dict) -> dict:
    """Main entrypoint. Routed to by the intent classifier on portfolio_health."""
    _validate_user_context(user_context)

    base_ccy = (user_context.get("base_currency") or "USD").upper()
    prefs = user_context.get("preferences") or {}
    benchmark = prefs.get("preferred_benchmark") or _default_benchmark(
        user_context.get("country", "US")
    )
    positions = user_context.get("positions") or []

    if not positions:
        return await _build_empty_portfolio_response(user_context, benchmark)

    async with _data_session() as yf:
        metrics = await compute_all(yf, positions, base_ccy, benchmark)

    observations = await _generate_observations(user_context, metrics)

    return _format_response(user_context, metrics, observations)


# --- empty portfolio (BUILD path) ------------------------------------------


async def _build_empty_portfolio_response(user_context: dict, benchmark: str) -> dict:
    """User has no positions yet. Skip MCP and produce a starter-oriented response."""
    client = AsyncOpenAI()
    system = (
        "You are a portfolio coach speaking to someone who has not yet "
        "started investing. Their account is verified and ready. Suggest "
        "1-3 concrete first considerations tailored to their age, country, "
        "risk profile, and base currency. Plain language, no jargon without "
        "a one-line context. Don't recommend specific tickers — frame in "
        "terms of building blocks (broad index fund, bond allocation, "
        "emergency fund, etc.). Output JSON: "
        '{"observations": [{"severity": "info", "text": "..."}]}'
    )
    user = json.dumps(
        {
            "name": user_context.get("name"),
            "age": user_context.get("age"),
            "country": user_context.get("country"),
            "base_currency": user_context.get("base_currency"),
            "risk_profile": user_context.get("risk_profile"),
        }
    )
    obs = await _call_llm_json(client, system, user)

    return {
        "user_id": user_context.get("user_id"),
        "name": user_context.get("name"),
        "status": "ready_to_build",
        "summary": (
            f"Your account is verified and you have no positions yet. "
            f"Here's how someone with a {user_context.get('risk_profile', 'moderate')} "
            f"risk profile in {user_context.get('country', 'your market')} typically gets started."
        ),
        "concentration_risk": None,
        "performance": None,
        "benchmark_comparison": {"benchmark": benchmark, "note": "no positions to compare"},
        "observations": obs.get("observations", []),
        "disclaimer": DISCLAIMER,
    }


# --- normal path -----------------------------------------------------------


async def _generate_observations(
    user_context: dict, metrics: PortfolioMetrics
) -> list[dict]:
    """Turn computed metrics into 1-3 plain-language observations."""
    client = AsyncOpenAI()

    system = (
        "You are a portfolio analyst writing for a novice investor. You'll "
        "receive computed metrics — DO NOT invent or recompute numbers; only "
        "interpret what's given. Surface the 1-3 things that matter MOST for "
        "this specific user, considering their age, risk profile, country, "
        "and any preferences (e.g. income_focus). Plain language. If you use "
        "a financial term, give a one-line context the same sentence.\n\n"
        "Important framing rules:\n"
        "  - A conservative or income-focused portfolio is EXPECTED to "
        "underperform broad equity indices like the S&P 500. Do NOT suggest "
        "restructuring on benchmark underperformance alone if the allocation "
        "matches the stated risk profile. Frame it as 'expected for your "
        "strategy' instead.\n"
        "  - High concentration is more concerning for older or conservative "
        "users than for young aggressive ones — calibrate severity accordingly.\n"
        "  - Don't recommend specific tickers. Talk in terms of categories "
        "(broad index, bonds, international exposure, etc.).\n\n"
        "Severity rules:\n"
        '  - "warning": something the user should likely act on (e.g. risk-'
        "profile mismatch, dangerous concentration, age-inappropriate allocation)\n"
        '  - "info": neutral observation\n'
        '  - "positive": something working well\n\n'
        "Output strictly JSON:\n"
        '{"observations": [{"severity": "...", "text": "..."}]}\n'
        "Maximum 3 observations. Order by importance."
    )

    user_payload = {
        "user": {
            "age": user_context.get("age"),
            "country": user_context.get("country"),
            "risk_profile": user_context.get("risk_profile"),
            "base_currency": user_context.get("base_currency"),
            "preferences": user_context.get("preferences", {}),
        },
        "metrics": _metrics_to_dict(metrics),
    }

    out = await _call_llm_json(client, system, json.dumps(user_payload, default=_json_default))
    obs = out.get("observations", [])
    return obs[:3]


async def _call_llm_json(client: AsyncOpenAI, system: str, user: str) -> dict:
    resp = await client.chat.completions.create(
        model=OBSERVATIONS_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {}


# --- formatting ------------------------------------------------------------


def _format_response(
    user_context: dict, metrics: PortfolioMetrics, observations: list[dict]
) -> dict:
    perf = metrics.performance
    conc = metrics.concentration_risk
    bench = metrics.benchmark_comparison

    return {
        "user_id": user_context.get("user_id"),
        "name": user_context.get("name"),
        "status": "ok",
        "base_currency": metrics.base_currency,
        "as_of": date.today().isoformat(),
        "holdings": [
            {
                "ticker": p.ticker,
                "quantity": p.quantity,
                "weight_pct": round(p.weight_pct or 0, 2),
                "market_value": round(p.market_value_base or 0, 2),
                "return_pct": round(p.return_pct or 0, 2),
                "native_currency": p.price_currency or p.cost_currency,
            }
            for p in metrics.positions
        ],
        "performance": {
            "total_value": perf.total_value_base,
            "total_cost": perf.total_cost_base,
            "total_return_pct": perf.total_return_pct,
            "annualized_return_pct": perf.annualized_return_pct,
            "weighted_holding_days": perf.period_days,
        },
        "concentration_risk": {
            "top_position_pct": conc.top_position_pct,
            "top_position_ticker": conc.largest_holding,
            "top_3_positions_pct": conc.top_3_positions_pct,
            "hhi": conc.hhi,
            "flag": conc.flag,
        },
        "benchmark_comparison": (
            {
                "benchmark": bench.benchmark,
                "portfolio_return_pct": bench.portfolio_return_pct,
                "benchmark_return_pct": bench.benchmark_return_pct,
                "alpha_pct": bench.alpha_pct,
                "period_start": bench.period_start.isoformat(),
                "period_end": bench.period_end.isoformat(),
            }
            if bench
            else None
        ),
        "fx_rates_used": metrics.fx_rates_used,
        "observations": observations,
        "disclaimer": DISCLAIMER,
    }


# --- helpers ---------------------------------------------------------------


def _validate_user_context(ctx: dict) -> None:
    if "user_id" not in ctx:
        raise ValueError("user_context is missing 'user_id'")
    if (ctx.get("kyc") or {}).get("status") != "verified":
        # Not fatal — but worth flagging upstream. Agent still runs.
        pass


def _default_benchmark(country: str) -> str:
    return {
        "US": "S&P 500",
        "GB": "FTSE 100",
        "SG": "MSCI World",
        "JP": "NIKKEI",
        "IN": "NIFTY 50",
    }.get((country or "US").upper(), "MSCI World")


def _metrics_to_dict(m: PortfolioMetrics) -> dict:
    return {
        "concentration_risk": asdict(m.concentration_risk),
        "performance": asdict(m.performance),
        "benchmark_comparison": (
            asdict(m.benchmark_comparison) if m.benchmark_comparison else None
        ),
        "positions": [
            {
                "ticker": p.ticker,
                "weight_pct": round(p.weight_pct or 0, 2),
                "return_pct": round(p.return_pct or 0, 2),
            }
            for p in m.positions
        ],
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, date):
        return o.isoformat()
    if is_dataclass(o):
        return asdict(o)
    raise TypeError(f"not serializable: {type(o)}")
