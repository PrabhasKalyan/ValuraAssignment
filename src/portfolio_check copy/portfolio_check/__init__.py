# Keep __init__ side-effect-free so submodules (e.g. metrics) can be imported
# without pulling in optional deps like openai. Use:
#     from portfolio_check.agent import run_portfolio_check
