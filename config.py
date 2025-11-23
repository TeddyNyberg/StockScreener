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

#Models
#CLOSE_ONLY_STATIC_PREFIX = 'pytorch-training-2025-10-03-15-34-22-625/output/model.tar.gz'
#CLOSE_ONLY_TUNED_PREFIX = "fine_tuned_models/only_close_model_original_not_fine_tuned.gz"


MODEL_MAP = {
    "A": {
        "name": "NYBERG STATIC MODEL",
        "csv_filepath": "nyberg_results_static.csv",
        "s3_prefix": "pytorch-training-2025-10-03-15-34-22-625/output/model.tar.gz",
        "prefix": "pytorch-training-2025-10-03-15-34-22-625/output/model.tar.gz",
        "model_filepath": "nyberg_static.pth"

    },
    "B": {
        "name": "NYBERG TUNED MODEL",
        "csv_filepath": "nyberg_results_tune.csv",
        "s3_prefix": "fine_tuned_models/only_close_model_original_not_fine_tuned.gz",
        "model_filepath": "tuned_model.pth"
    },
    "C":{
        "name": "NYBERG STATIC QUANTIZED MODEL",
        "csv_filepath": "nyberg_results_static_quantized.csv",
        "s3_prefix": "pytorch-training-2025-10-03-15-34-22-625/output/model.tar.gz",
        "model_filepath": "nyberg_static.pth"
    }
}