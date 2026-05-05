"""
CLI runner: `python -m portfolio_check usr_001`

Loads user_context from users.json, runs the agent, prints JSON.
For ad-hoc input: `python -m portfolio_check - < context.json`
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

from .agent import run_portfolio_check


def _json_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if is_dataclass(o):
        return asdict(o)
    raise TypeError(f"not serializable: {type(o)}")


async def _main() -> None:
    load_dotenv()
    if len(sys.argv) < 2:
        print("usage: python -m portfolio_check <user_id|->", file=sys.stderr)
        sys.exit(2)

    arg = sys.argv[1]
    if arg == "-":
        ctx = json.load(sys.stdin)
    else:
        users = json.loads((Path(__file__).parent / "users.json").read_text())
        if arg not in users:
            print(f"unknown user_id: {arg}; known: {list(users)}", file=sys.stderr)
            sys.exit(2)
        ctx = users[arg]

    result = await run_portfolio_check(ctx)
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    asyncio.run(_main())
