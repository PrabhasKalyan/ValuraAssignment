"""
Skeleton test for classifier routing accuracy on the labeled gold set.

Wire your classifier import and remove the @pytest.mark.skip decorator.
The success threshold (≥ 85%) is from ASSIGNMENT.md.

This test demonstrates the entity matcher pattern. The matcher rules are in
fixtures/README.md — follow them or document any deviations in your README.
"""
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Entity matcher — implements the rules in fixtures/README.md
# ---------------------------------------------------------------------------
#
# This is a STARTER matcher. It covers the most common cases (tickers, topics,
# amounts, rates, generic exact-match). Before relying on it for grading, you
# must extend it to cover the full vocabulary in
# fixtures/test_queries/intent_classification.json → entity_vocabulary:
#
#   - period_years      — exact integer match
#   - currency          — ISO 4217 exact
#   - frequency         — vocabulary token (daily/weekly/monthly/yearly)
#   - horizon           — vocabulary token (6_months / 1_year / 5_years / ...)
#   - time_period       — vocabulary token (today / this_week / this_month / ...)
#   - index             — exact match against canonical names (S&P 500, FTSE 100, ...)
#   - action            — vocabulary token (buy / sell / hold / hedge / rebalance)
#   - goal              — vocabulary token (retirement / education / house / FIRE / ...)
#
# The "else" branch below catches all of these via lowercase string comparison,
# which is correct for vocabulary tokens but NOT correct for `index` (e.g. "S&P 500"
# should be case-sensitive on letters but tolerant of "S&P500" vs "S&P 500" spacing).
# Extend deliberately — document any deviation in your README.

def _normalize_ticker(t: str) -> str:
    """Case-fold and drop the exchange suffix (AAPL.US → AAPL)."""
    return t.upper().split(".")[0]


def matches_entities(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """
    Subset match with normalization. `actual` must contain every value in
    `expected`; extra fields and extra values are allowed.

    Extend this for the full entity_vocabulary — see comment above.
    """
    for field, exp_value in expected.items():
        act_value = actual.get(field)
        if act_value is None:
            return False

        if field == "tickers":
            exp_set = {_normalize_ticker(t) for t in exp_value}
            act_set = {_normalize_ticker(t) for t in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("topics", "sectors"):
            exp_set = {s.lower() for s in exp_value}
            act_set = {s.lower() for s in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("amount", "rate"):
            if abs(act_value - exp_value) > abs(exp_value) * 0.05:
                return False
        elif field == "period_years":
            if int(act_value) != int(exp_value):
                return False
        else:
            # Catch-all for vocabulary tokens (action, goal, frequency, horizon,
            # time_period, currency, index). Override per-field if you need more
            # nuanced normalization (e.g. spacing-tolerant index matching).
            if str(act_value).lower() != str(exp_value).lower():
                return False
    return True


# ---------------------------------------------------------------------------
# Routing accuracy — this is the test we score
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Stub — wire up your classifier import below and remove this decorator")
def test_classifier_routing_accuracy(gold_classifier_queries, mock_llm):
    """
    Threshold: ≥ 85% routing accuracy.
    """
    # from src.classifier import classify  # noqa: ERA001

    correct = 0
    for case in gold_classifier_queries:
        result = classify(case["query"], llm=mock_llm)  # noqa: F821
        if result.agent == case["expected_agent"]:
            correct += 1

    accuracy = correct / len(gold_classifier_queries)
    assert accuracy >= 0.85, f"Routing accuracy {accuracy:.2%} below 85%"


@pytest.mark.skip(reason="Stub — wire up your classifier import below and remove this decorator")
def test_classifier_entity_extraction(gold_classifier_queries, mock_llm):
    """
    Soft signal — not a hard threshold. Reported, not failed on.
    """
    matched = 0
    total_with_entities = 0
    for case in gold_classifier_queries:
        if not case["expected_entities"]:
            continue
        total_with_entities += 1
        result = classify(case["query"], llm=mock_llm)  # noqa: F821
        if matches_entities(result.entities, case["expected_entities"]):
            matched += 1

    # No assertion — emit a report
    rate = matched / total_with_entities if total_with_entities else 0.0
    print(f"\nEntity match rate: {rate:.2%} ({matched}/{total_with_entities})")
