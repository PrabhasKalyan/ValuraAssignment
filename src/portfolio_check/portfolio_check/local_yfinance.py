"""
In-process yfinance adapter — same interface as the MCP YFinanceClient.

The production path is mcp_client.YFinanceClient (stdio MCP server). This
file is a fallback for environments where spinning up an MCP server is
inconvenient (e.g. local tests, no `uv`/`uvx`). Toggle via:

    PORTFOLIO_DATA_BACKEND=local   # use this module
    PORTFOLIO_DATA_BACKEND=mcp     # default
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import AsyncIterator

import yfinance as yf

from .mcp_client import HistoryPoint, Quote, normalize_currency


@asynccontextmanager
async def local_session() -> AsyncIterator["LocalYFinanceClient"]:
    yield LocalYFinanceClient()


class LocalYFinanceClient:
    """Mirror of mcp_client.YFinanceClient, backed by the yfinance package."""

    async def get_quote(self, ticker: str) -> Quote | None:
        return await asyncio.to_thread(self._sync_quote, ticker)

    def _sync_quote(self, ticker: str) -> Quote | None:
        try:
            t = yf.Ticker(ticker)
            info = getattr(t, "fast_info", None)
            price = None
            currency = None
            if info is not None:
                price = info.get("last_price") if hasattr(info, "get") else getattr(info, "last_price", None)
                currency = info.get("currency") if hasattr(info, "get") else getattr(info, "currency", None)
            if price is None:
                # Fall back to last close from a 1-day history.
                hist = t.history(period="5d", auto_adjust=False)
                if hist is None or hist.empty:
                    return None
                price = float(hist["Close"].iloc[-1])
                currency = currency or "USD"
            if currency is None:
                currency = "USD"
            price, currency = normalize_currency(float(price), str(currency))
            return Quote(ticker=ticker, price=price, currency=currency)
        except Exception:
            return None

    async def get_history(
        self, ticker: str, start: date, end: date | None = None
    ) -> list[HistoryPoint]:
        return await asyncio.to_thread(self._sync_history, ticker, start, end)

    def _sync_history(
        self, ticker: str, start: date, end: date | None
    ) -> list[HistoryPoint]:
        end = end or date.today()
        try:
            df = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
            )
        except Exception:
            return []
        if df is None or df.empty:
            return []
        out: list[HistoryPoint] = []
        for idx, row in df.iterrows():
            d = idx.date() if hasattr(idx, "date") else idx
            out.append(HistoryPoint(date=d, close=float(row["Close"])))
        return out

    async def get_fx_rate(self, from_ccy: str, to_ccy: str) -> float | None:
        if from_ccy == to_ccy:
            return 1.0
        q = await self.get_quote(f"{from_ccy}{to_ccy}=X")
        return q.price if q else None
