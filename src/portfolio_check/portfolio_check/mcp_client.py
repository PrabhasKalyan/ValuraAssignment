"""
Thin async wrapper around a yfinance MCP server (stdio transport).

Tested against the `yfmcp` / `yfinance-mcp` family of servers, which expose
tools like `get_ticker_info` and `get_history`. If you swap to a different
yfinance MCP server, only the small adapter methods at the bottom need to
change — the agent treats this module as the single integration point.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from mcp import ClientSession, StdioServerParameters


@dataclass
class Quote:
    ticker: str
    price: float
    currency: str


# Yahoo Finance returns LSE prices in pence (`GBp`), JSE in cents (`ZAc`),
# TASE in agorot (`ILa`) — note the lowercase last letter. Cost-basis in
# users.json is in major units, so without this normalization, returns blow
# up by 100x. The mixed-case code is the actual signal Yahoo uses.
_MINOR_UNIT_CODES = {
    "GBp": ("GBP", 100.0),
    "GBX": ("GBP", 100.0),
    "ZAc": ("ZAR", 100.0),
    "ILa": ("ILS", 100.0),
}


def normalize_currency(price: float, currency: str) -> tuple[float, str]:
    """If `currency` is a minor unit (e.g. GBp), convert to major (GBP)."""
    if currency in _MINOR_UNIT_CODES:
        major, factor = _MINOR_UNIT_CODES[currency]
        return price / factor, major
    return price, currency.upper()


@dataclass
class HistoryPoint:
    date: date
    close: float


def _server_params() -> "StdioServerParameters":
    """Spawn a yfinance MCP server.

    Defaults to `uvx yfmcp@latest` (https://pypi.org/project/yfmcp/) which
    is a zero-install option. Override with PORTFOLIO_MCP_CMD /
    PORTFOLIO_MCP_ARGS if you host your own."""
    from mcp import StdioServerParameters

    cmd = os.getenv("PORTFOLIO_MCP_CMD", "uvx")
    args_env = os.getenv("PORTFOLIO_MCP_ARGS", "yfmcp@latest")
    return StdioServerParameters(command=cmd, args=args_env.split())


@asynccontextmanager
async def yfinance_session() -> AsyncIterator["YFinanceClient"]:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield YFinanceClient(session)


def _extract_text(result: Any) -> str:
    """MCP tool results come back as a list of content parts; we want text."""
    parts = getattr(result, "content", None) or []
    chunks: list[str] = []
    for p in parts:
        text = getattr(p, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _parse_json(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class YFinanceClient:
    """Adapter over the MCP session. Keep tool-specific assumptions here."""

    def __init__(self, session: ClientSession):
        self._session = session

    async def _call(self, tool: str, args: dict) -> Any:
        result = await self._session.call_tool(tool, args)
        return _parse_json(_extract_text(result))

    # --- public API used by the agent --------------------------------------

    async def get_quote(self, ticker: str) -> Quote | None:
        """Latest price + native currency for a ticker."""
        info = await self._call("get_ticker_info", {"symbol": ticker})
        if not isinstance(info, dict):
            return None
        price = (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )
        currency = info.get("currency") or "USD"
        if price is None:
            return None
        price, currency = normalize_currency(float(price), str(currency))
        return Quote(ticker=ticker, price=price, currency=currency)

    async def get_history(
        self, ticker: str, start: date, end: date | None = None
    ) -> list[HistoryPoint]:
        """Daily close history. Used for benchmark comparison."""
        end = end or date.today()
        # Choose a period bucket that covers `start` — yfinance accepts both
        # period strings and explicit date ranges; we send dates for accuracy.
        raw = await self._call(
            "get_history",
            {
                "symbol": ticker,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "interval": "1d",
            },
        )
        return _coerce_history(raw)

    async def get_fx_rate(self, from_ccy: str, to_ccy: str) -> float | None:
        """Spot FX. Uses Yahoo's `XXXYYY=X` convention."""
        if from_ccy == to_ccy:
            return 1.0
        symbol = f"{from_ccy}{to_ccy}=X"
        q = await self.get_quote(symbol)
        return q.price if q else None


def _coerce_history(raw: Any) -> list[HistoryPoint]:
    """yfinance MCP servers return history in different shapes — normalize."""
    if raw is None:
        return []
    points: list[HistoryPoint] = []

    # Shape 1: list[ {date, close, ...} ]
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            d = _parse_date(row.get("date") or row.get("Date"))
            close = row.get("close") or row.get("Close")
            if d and close is not None:
                points.append(HistoryPoint(date=d, close=float(close)))
        return points

    # Shape 2: dict keyed by date -> {close, ...}
    if isinstance(raw, dict):
        # Some servers wrap under a key
        for key in ("data", "history", "prices"):
            if key in raw and isinstance(raw[key], (list, dict)):
                return _coerce_history(raw[key])
        for k, v in raw.items():
            d = _parse_date(k)
            if not d or not isinstance(v, dict):
                continue
            close = v.get("close") or v.get("Close")
            if close is not None:
                points.append(HistoryPoint(date=d, close=float(close)))
        points.sort(key=lambda p: p.date)
        return points

    return points


def _parse_date(s: Any) -> date | None:
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if isinstance(s, datetime):
        return s.date()
    if not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[: len(fmt) + 4], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def trading_days_ago(days: int) -> date:
    return date.today() - timedelta(days=days)
