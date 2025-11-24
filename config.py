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

SP_TICKERS_TESTING = ['NVDA', 'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'AVGO', 'META', 'TSLA', 'BRK.B', 'LLY', 'WMT', 'JPM', 'V', 'ORCL', 'XOM', 'JNJ', 'MA', 'NFLX', 'ABBV', 'COST', 'BAC', 'PLTR', 'PG', 'HD', 'AMD', 'KO', 'GE', 'CVX', 'CSCO', 'UNH', 'IBM', 'WFC', 'CAT', 'MS', 'AXP', 'GS', 'MRK', 'PM', 'TMUS', 'MU', 'RTX', 'ABT', 'TMO', 'MCD', 'CRM', 'PEP', 'ISRG', 'LIN', 'DIS', 'INTU', 'T', 'AMGN', 'LRCX', 'AMAT', 'C', 'APP', 'BX', 'QCOM', 'UBER', 'NEE', 'VZ', 'NOW', 'TJX', 'BLK', 'INTC', 'APH', 'SCHW', 'DHR', 'GILD', 'ACN', 'BKNG', 'GEV', 'SPGI', 'ANET', 'TXN', 'KLAC', 'BSX', 'PFE', 'SYK', 'WELL', 'BA', 'ADBE', 'UNP', 'PGR', 'COF', 'DE', 'LOW', 'MDT', 'ETN', 'PANW', 'CRWD', 'HON', 'PLD', 'CB', 'ADI', 'HCA', 'VRTX', 'COP', 'MCK', 'LMT', 'PH', 'KKR', 'CEG', 'ADP', 'CMCSA', 'CVS', 'CME', 'SO', 'MO', 'SBUX', 'HOOD', 'DUK', 'BMY', 'NKE', 'GD', 'NEM', 'TT', 'MMM', 'MMC', 'ICE', 'WM', 'MCO', 'ORLY', 'AMT', 'SHW', 'DELL', 'CDNS', 'DASH', 'NOC', 'UPS', 'MAR', 'HWM', 'REGN', 'TDG', 'ECL', 'APO', 'CTAS', 'AON', 'CI', 'USB', 'BK', 'EQIX', 'MDLZ', 'PNC', 'WMB', 'SNPS', 'EMR', 'RCL', 'ITW', 'ELV', 'COR', 'MNST', 'JCI', 'ABNB', 'SPG', 'GLW', 'RSG', 'GM', 'CL', 'CMI', 'AZO', 'COIN', 'TRV', 'AJG', 'AEP', 'TEL', 'NSC', 'PWR', 'CSX', 'HLT', 'FDX', 'ADSK', 'MSI', 'SRE', 'WDAY', 'KMI', 'FTNT', 'TFC', 'AFL', 'EOG', 'IDXX', 'WBD', 'MPC', 'APD', 'FCX', 'VST', 'PYPL', 'ROST', 'ALL', 'DDOG', 'BDX', 'DLR', 'PCAR', 'SLB', 'PSX', 'ZTS', 'VLO', 'D', 'O', 'LHX', 'STX', 'F', 'URI', 'NDAQ', 'EA', 'CAH', 'MET', 'EW', 'BKR', 'NXPI', 'ROP', 'WDC', 'PSA', 'XEL', 'EXC', 'CBRE', 'FAST', 'GWW', 'AME', 'OKE', 'CTVA', 'CARR', 'KR', 'TTWO', 'LVS', 'A', 'DHI', 'ROK', 'YUM', 'FICO', 'ETR', 'MSCI', 'FANG', 'CMG', 'MPWR', 'AMP', 'AXON', 'AIG', 'OXY', 'PEG', 'PAYX', 'TGT', 'CPRT', 'CCI', 'IQV', 'VMC', 'HIG', 'DAL', 'HSY', 'KDP', 'XYZ', 'VTR', 'PRU', 'GRMN', 'SYY', 'CTSH', 'TRGP', 'RMD', 'EBAY', 'MLM', 'WEC', 'ED', 'EQT', 'KMB', 'CCL', 'NUE', 'GEHC', 'TKO', 'PCG', 'OTIS', 'WAB', 'XYL', 'ACGL', 'FIS', 'FISV', 'EL', 'STT', 'KVUE', 'LEN', 'VRSK', 'IR', 'VICI', 'NRG', 'LYV', 'EXPE', 'RJF', 'WTW', 'KHC', 'UAL', 'KEYS', 'WRB', 'MTD', 'CHTR', 'FOXA', 'EXR', 'K', 'ROL', 'MTB', 'CSGP', 'ATO', 'AEE', 'DTE', 'ADM', 'ODFL', 'FITB', 'TSCO', 'FOX', 'MCHP', 'BRO', 'EXE', 'HUM', 'IBKR', 'FE', 'HPE', 'SYF', 'FSLR', 'PPL', 'BR', 'CBOE', 'EFX', 'EME', 'CINF', 'AWK', 'STE', 'CNP', 'GIS', 'BIIB', 'AVB', 'DOV', 'IRM', 'HBAN', 'TER', 'VLTO', 'ES', 'NTRS', 'LDOS', 'EQR', 'DXCM', 'WAT', 'PHM', 'VRSN', 'PODD', 'STZ', 'TDY', 'ULTA', 'STLD', 'EIX', 'CMS', 'CFG', 'HUBB', 'HPQ', 'DG', 'DVN', 'PPG', 'LH', 'L', 'TROW', 'RF', 'HAL', 'TPR', 'WSM', 'NTAP', 'DGX', 'JBL', 'NVR', 'SBAC', 'RL', 'TPL', 'PTC', 'DLTR', 'NI', 'TYL', 'DRI', 'CPAY', 'CHD', 'INCY', 'LULU', 'IP', 'CTRA', 'AMCR', 'WST', 'KEY', 'SMCI', 'EXPD', 'TTD', 'TSN', 'ON', 'PFG', 'TRMB', 'MKC', 'BG', 'ZBH', 'CDW', 'CNC', 'CHRW', 'GPC', 'PKG', 'SW', 'LNT', 'SNA', 'EVRG', 'PSKY', 'ESS', 'GPN', 'INVH', 'IFF', 'GDDY', 'PNR', 'LUV', 'IT', 'FTV', 'HOLX', 'GEN', 'LII', 'DD', 'BBY', 'MAA', 'APTV', 'Q', 'JBHT', 'DOW', 'WY', 'ERIE', 'J', 'NWS', 'COO', 'UHS', 'OMC', 'LYB', 'NWSA', 'SOLV', 'TXT', 'ALLE', 'KIM', 'DPZ', 'ALB', 'FFIV', 'BF.B', 'BALL', 'REG', 'AVY', 'NDSN', 'EG', 'MAS', 'UDR', 'AKAM', 'IEX', 'CLX', 'DOC', 'HRL', 'DECK', 'BXP', 'JKHY', 'WYNN', 'CF', 'ZBRA', 'HST', 'VTRS', 'HII', 'CPT', 'AIZ', 'BEN', 'SJM', 'BLDR', 'RVTY', 'HAS', 'DAY', 'PNW', 'GL', 'FDS', 'IVZ', 'SWK', 'ALGN', 'EPAM', 'AES', 'TECH', 'CPB', 'BAX', 'IPG', 'SWKS', 'MRNA', 'TAP', 'AOS', 'POOL', 'MGM', 'PAYC', 'ARE', 'HSIC', 'GNRC', 'FRT', 'CAG', 'APA', 'DVA', 'NCLH', 'CRL', 'LW', 'MOS', 'MTCH', 'LKQ', 'MOH', 'SOLS', 'MHK']
