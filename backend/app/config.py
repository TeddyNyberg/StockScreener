
from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

#S&P ticker list location
S_AND_P_URL = "https://stockanalysis.com/list/sp-500-stocks/"
SP_TICKER_CACHE = DATA_DIR / "sp500_tickers.csv"
SP_DATA_CACHE = DATA_DIR / "sp500_history_cache.parquet"
VOL_DATA_CACHE = DATA_DIR / "vol_data.npy"

BACKTEST_REPORT_PATH = DATA_DIR / "backtest_portfolio_PL.xlsx" # Centralize this too

MODEL_MAP = {
    "A": {
        "name": "NYBERG STATIC MODEL",
        "csv_filepath": DATA_DIR / "nyberg_results_static.csv",
        "s3_prefix": "pytorch-training-2025-10-03-15-34-22-625/output/model.tar.gz",
        "model_filepath": DATA_DIR / "nyberg_static.pth", # Move models to data or a 'models' folder
        "quantized": False
    },
    "B": {
        "name": "NYBERG TUNED MODEL",
        "csv_filepath": DATA_DIR / "nyberg_results_tune.csv",
        "s3_prefix": "fine_tuned_models/only_close_model_original_not_fine_tuned.gz",
        "model_filepath": DATA_DIR / "tuned_model.pth",
        "quantized": False
    },
    "C":{
        "name": "NYBERG STATIC QUANTIZED MODEL",
        "csv_filepath": DATA_DIR / "nyberg_results_static_quantized.csv",
        "s3_prefix": "pytorch-training-2025-10-03-15-34-22-625/output/model.tar.gz",
        "model_filepath": DATA_DIR / "nyberg_static.pth",
        "quantized": True
    },
    "D":{
        "name": "NYBERG STATIC MODEL FULL KELLY",
        "csv_filepath": DATA_DIR / "nyberg_results_static_full_kelly.csv",
        "s3_prefix": "pytorch-training-2025-10-03-15-34-22-625/output/model.tar.gz",
        "model_filepath": DATA_DIR / "nyberg_static.pth",
        "quantized": False
    }
}