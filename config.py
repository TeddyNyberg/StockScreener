# Trading Strat
RISK_FREE_RATE = 0.005      # 0.5% annualized
HALF_KELLY_MULTIPLIER = 1   # 1 for full Kelly, 0.5 for half Kelly
transaction_cost_pct = 0.0000 # eventually turn to 5bp, 0.5%
initial_capital_fully = 100000.0
lookback_period = "6M"

# Model Parameters
SEQUENCE_SIZE = 50

# Charting/Data Defaults
DEFAULT_CHART_TIME = "1Y"
DEFAULT_STYLE = "charles"

#S&P ticker list location
S_AND_P_URL = "https://stockanalysis.com/list/sp-500-stocks/"


MODEL_MAP = {
    "A": {
        "name": "NYBERG STATIC MODEL",
        "filepath": "nyberg_results_static.csv",
        "prefix": "pytorch-training-2025-10-03-15-34-22-625/output/model.tar.gz"
    },
    "B": {
        "name": "NYBERG TUNED MODEL",
        "filepath": "nyberg_results_tune.csv",
        "prefix": "fine_tuned_models/only_close_model_original_not_fine_tuned.gz"
    },
    "C":{
        "name": "NYBERG STATIC QUANTIZED MODEL",
        "filepath": "nyberg_results_static_quantized.csv",
        "prefix": "fine_tuned_models/close_only_quantized_static.gz"
    }
}