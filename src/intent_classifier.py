from openai import OpenAI
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()


MODEL_DEV = "gpt-4.0-mini"
MODEL_TEST = "gpt-4.1"

MAX_INPUT_TOKENS = 800
MAX_OUTPUT_TOKENS = 150

INTENT_DEFINITIONS = {
    "portfolio_health": "portfolio analysis performance diversification holdings allocation",
    "market_research": "info news prices markets stocks indices sectors quotes",
    "investment_strategy": "buy sell hold rebalance hedge advice trade position",
    "financial_planning": "long-term goals retirement house education FIRE savings plan",
    "financial_calculator": "numeric computations returns mortgage tax FX compute calculate",
    "risk_assessment": "risk metrics drawdown stress testing exposure volatility VaR",
    "product_recommendation": "recommend funds ETFs products mutual fund suggest",
    "predictive_analysis": "forecast forecasts future predictions outlook projection expected next year",
    "customer_support": "account help how-to issues login password reset support",
    "general_query": "definition explanation greeting hello hi unknown meaning",
}

_INTENT_LABELS = list(INTENT_DEFINITIONS.keys())
_vectorizer = TfidfVectorizer(stop_words="english")
_intent_matrix = _vectorizer.fit_transform(INTENT_DEFINITIONS.values())

def _tfidf_fallback(query: str):
    query_vec = _vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _intent_matrix)[0]
    best_idx = int(sims.argmax())
    if sims[best_idx] < 0.05:
        return {"agent": "general_query", "entities": {}}
    return {"agent": _INTENT_LABELS[best_idx], "entities": {}}


def intent_classifier(user_context: str, query: str, testing: bool = False):
    model = MODEL_TEST if testing else MODEL_DEV

    system_prompt = """
You are an intent classifier for a financial assistant.

Classify the query into EXACTLY one of:
portfolio_health, market_research, investment_strategy,
financial_planning, financial_calculator, risk_assessment,
product_recommendation, predictive_analysis,
customer_support, general_query.

Definitions:
- portfolio_health: portfolio analysis, performance, diversification
- market_research: info/news/prices about markets, stocks, indices
- investment_strategy: buy/sell/hold/rebalance/hedge advice
- financial_planning: long-term goals (retirement, house, education, FIRE)
- financial_calculator: numeric computations (returns, mortgage, tax, FX)
- risk_assessment: risk metrics, drawdown, stress testing, exposure
- product_recommendation: recommend funds/ETFs/products
- predictive_analysis: forecasts, future predictions
- customer_support: account/help/how-to issues
- general_query: definitions, explanations, greetings, unknown

Also extract entities (only if present):
- tickers (uppercase, include suffix if relevant)
- amount (number)
- currency (ISO 4217)
- rate (decimal)
- period_years (integer)
- frequency (daily, weekly, monthly, yearly)
- horizon (6_months, 1_year, 5_years)
- time_period (today, this_week, this_month, this_year)
- topics (array of strings)
- sectors (array of strings)
- index (string)
- action (buy, sell, hold, hedge, rebalance)
- goal (retirement, education, house, FIRE, emergency_fund)

Rules:
- Return ONLY valid JSON
- No explanation, no extra text
- If unclear → general_query
- Entities must be minimal and normalized
"""

    user_prompt = f"""
query: {query}
user_context: {user_context}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ]
        )
        raw_output = response.choices[0].message.content
        return json.loads(raw_output)
    except Exception:
        return _tfidf_fallback(query)
