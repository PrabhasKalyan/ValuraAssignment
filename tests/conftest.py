"""
Shared pytest fixtures for the Valura AI assignment.

The most important fixture here is `mock_llm` — every test that touches the
classifier or any LLM-using code must use it. CI runs without OPENAI_API_KEY
and unmocked LLM calls will fail.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture loaders
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def load_user():
    """Load a user fixture by id, e.g. load_user('usr_001')."""
    def _load(user_id: str) -> dict:
        for path in (FIXTURES_DIR / "users").glob("*.json"):
            with open(path, encoding="utf-8") as f:
                user = json.load(f)
            if user["user_id"] == user_id:
                return user
        raise FileNotFoundError(f"No fixture for user {user_id}")
    return _load


@pytest.fixture
def gold_classifier_queries() -> list[dict]:
    with open(FIXTURES_DIR / "test_queries" / "intent_classification.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture
def gold_safety_queries() -> list[dict]:
    with open(FIXTURES_DIR / "test_queries" / "safety_pairs.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture
def conversation_test_cases():
    """Returns a callable: conversation_test_cases('follow_up_session')."""
    def _load(name: str) -> list[dict]:
        path = FIXTURES_DIR / "conversations" / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)["test_cases"]
    return _load


# ---------------------------------------------------------------------------
# LLM mocking
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """
    Returns a MagicMock that you should configure per-test to return whatever
    structured output your classifier expects.

    Usage:
        def test_something(mock_llm):
            mock_llm.return_value = {"agent": "portfolio_health", "entities": {}}
            ...
    """
    return MagicMock()
