"""
Deterministic portfolio metrics. No LLM here — these numbers must be exact
and reproducible. The agent layer adds plain-language interpretation on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .mcp_client import YFinanceClient


# Map preferred-benchmark labels (as users write them) to actual yfinance
# tickers. Falls through to whatever the user typed if not in this table.
BENCHMARK_TICKERS = {
    "S&P 500": "^GSPC",
    "SP500": "^GSPC",
    "QQQ": "QQQ",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "MSCI WORLD": "URTH",
    "FTSE 100": "^FTSE",
    "STI": "^STI",
    "NIFTY 50": "^NSEI",
    "NIKKEI": "^N225",
}


@dataclass
class Position:
    ticker: str
    quantity: float
    avg_cost: float
    cost_currency: str
    purchased_at: date
    # Filled in by enrich_positions:
    current_price: float | None = None
    price_currency: str | None = None
    market_value_base: float | None = None
    cost_basis_base: float | None = None
    weight_pct: float | None = None
    return_pct: float | None = None


@dataclass
class ConcentrationRisk:
    top_position_pct: float
    top_3_positions_pct: float
    hhi: float  # Herfindahl-Hirschman: sum of weight_i^2 (in %^2 / 100)
    flag: str  # "low" | "moderate" | "high"
    largest_holding: str | None


@dataclass
class Performance:
    total_return_pct: float
    annualized_return_pct: float
    period_days: int
    total_value_base: float
    total_cost_base: float


@dataclass
class BenchmarkComparison:
    benchmark: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    period_start: date
    period_end: date


@dataclass
class PortfolioMetrics:
    base_currency: str
    positions: list[Position]
    concentration_risk: ConcentrationRisk
    performance: Performance
    benchmark_comparison: BenchmarkComparison | None
    fx_rates_used: dict[str, float] = field(default_factory=dict)


# --- enrichment -------------------------------------------------------------


def parse_position(p: dict) -> Position:
    return Position(
        ticker=p["ticker"],
        quantity=float(p["quantity"]),
        avg_cost=float(p["avg_cost"]),
        cost_currency=str(p.get("currency", "USD")).upper(),
        purchased_at=_parse_iso_date(p["purchased_at"]),
    )


def _parse_iso_date(s: str) -> date:
    return datetime.fromisoformat(s).date()


async def enrich_positions(
    yf: YFinanceClient,
    positions: list[Position],
    base_currency: str,
) -> dict[str, float]:
    """Fetch live quotes + FX. Mutates positions in place. Returns FX rates used."""
    fx_cache: dict[str, float] = {}

    async def fx_to_base(ccy: str) -> float:
        ccy = ccy.upper()
        if ccy == base_currency:
            return 1.0
        if ccy in fx_cache:
            return fx_cache[ccy]
        rate = await yf.get_fx_rate(ccy, base_currency)
        if rate is None:
            # Hard fail-safe: assume parity rather than crashing the whole run.
            rate = 1.0
        fx_cache[ccy] = rate
        return rate

    for pos in positions:
        quote = await yf.get_quote(pos.ticker)
        if quote is None:
            # No live price → use cost basis (so position still appears, no return)
            pos.current_price = pos.avg_cost
            pos.price_currency = pos.cost_currency
        else:
            pos.current_price = quote.price
            pos.price_currency = quote.currency

        price_fx = await fx_to_base(pos.price_currency or pos.cost_currency)
        cost_fx = await fx_to_base(pos.cost_currency)

        pos.market_value_base = pos.quantity * pos.current_price * price_fx
        pos.cost_basis_base = pos.quantity * pos.avg_cost * cost_fx
        if pos.cost_basis_base > 0:
            pos.return_pct = (pos.market_value_base / pos.cost_basis_base - 1) * 100
        else:
            pos.return_pct = 0.0

    total_value = sum(p.market_value_base or 0 for p in positions) or 1.0
    for pos in positions:
        pos.weight_pct = (pos.market_value_base or 0) / total_value * 100

    return {f"{ccy}->{base_currency}": rate for ccy, rate in fx_cache.items()}


# --- metric computations ---------------------------------------------------


def compute_concentration(positions: list[Position]) -> ConcentrationRisk:
    weights = sorted([p.weight_pct or 0 for p in positions], reverse=True)
    largest = max(positions, key=lambda p: p.weight_pct or 0, default=None)
    top1 = weights[0] if weights else 0.0
    top3 = sum(weights[:3])
    hhi = sum((w / 100) ** 2 for w in weights) * 10000  # 0..10000 scale

    if top1 >= 40 or hhi >= 2500:
        flag = "high"
    elif top1 >= 25 or hhi >= 1500:
        flag = "moderate"
    else:
        flag = "low"

    return ConcentrationRisk(
        top_position_pct=round(top1, 2),
        top_3_positions_pct=round(top3, 2),
        hhi=round(hhi, 2),
        flag=flag,
        largest_holding=largest.ticker if largest else None,
    )


def compute_performance(positions: list[Position]) -> Performance:
    total_value = sum(p.market_value_base or 0 for p in positions)
    total_cost = sum(p.cost_basis_base or 0 for p in positions)
    if total_cost <= 0:
        return Performance(0.0, 0.0, 0, total_value, total_cost)

    total_return = (total_value / total_cost - 1) * 100

    # Use weighted-average holding period for annualization. A simple
    # earliest-purchase approach overstates; weighting by cost is closer to
    # money-weighted return without requiring full cashflow history.
    today = date.today()
    weighted_days = sum(
        (today - p.purchased_at).days * (p.cost_basis_base or 0) for p in positions
    ) / total_cost
    weighted_days = max(weighted_days, 1)
    years = weighted_days / 365.25

    growth = total_value / total_cost
    annualized = (growth ** (1 / years) - 1) * 100 if years > 0 else 0.0

    return Performance(
        total_return_pct=round(total_return, 2),
        annualized_return_pct=round(annualized, 2),
        period_days=int(weighted_days),
        total_value_base=round(total_value, 2),
        total_cost_base=round(total_cost, 2),
    )


def resolve_benchmark_ticker(label: str) -> str:
    return BENCHMARK_TICKERS.get(label.upper(), label)


async def compute_benchmark_comparison(
    yf: YFinanceClient,
    positions: list[Position],
    benchmark_label: str,
    portfolio_return_pct: float,
) -> BenchmarkComparison | None:
    """Compare portfolio return vs. benchmark over the same period.

    Period = earliest purchase date in the portfolio → today. This isn't a
    perfect apples-to-apples (positions were added at different times), but
    it's the convention novices expect: "what would I have made in the index
    over the same window?"
    """
    if not positions:
        return None

    start = min(p.purchased_at for p in positions)
    end = date.today()
    ticker = resolve_benchmark_ticker(benchmark_label)

    history = await yf.get_history(ticker, start=start, end=end)
    if len(history) < 2:
        return None

    bench_start = history[0].close
    bench_end = history[-1].close
    bench_return = (bench_end / bench_start - 1) * 100

    return BenchmarkComparison(
        benchmark=benchmark_label,
        portfolio_return_pct=round(portfolio_return_pct, 2),
        benchmark_return_pct=round(bench_return, 2),
        alpha_pct=round(portfolio_return_pct - bench_return, 2),
        period_start=start,
        period_end=end,
    )


# --- top-level orchestration -----------------------------------------------


async def compute_all(
    yf: YFinanceClient,
    raw_positions: Iterable[dict],
    base_currency: str,
    benchmark_label: str,
) -> PortfolioMetrics:
    positions = [parse_position(p) for p in raw_positions]
    fx = await enrich_positions(yf, positions, base_currency)
    concentration = compute_concentration(positions)
    performance = compute_performance(positions)
    benchmark = await compute_benchmark_comparison(
        yf, positions, benchmark_label, performance.total_return_pct
    )
    return PortfolioMetrics(
        base_currency=base_currency,
        positions=positions,
        concentration_risk=concentration,
        performance=performance,
        benchmark_comparison=benchmark,
        fx_rates_used=fx,
    )
