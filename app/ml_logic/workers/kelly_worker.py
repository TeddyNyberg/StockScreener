import math
from config import RISK_FREE_RATE, HALF_KELLY_MULTIPLIER

def kelly_worker(triple):

    ticker, mu, sigma_squared = triple

    if math.isnan(mu):
        print(f"{ticker} skipped mu is nan")
        return None
    if sigma_squared <= 0 or math.isnan(sigma_squared):
        print(f"{ticker} skipped sig_sq is nan")
        return None
    try:
        kelly_fraction = (mu - RISK_FREE_RATE) / sigma_squared
        allocation = kelly_fraction * HALF_KELLY_MULTIPLIER

        allocation = max(0.0, min(1.0, allocation))
        if math.isnan(allocation) or math.isinf(allocation):
            print(f"{ticker} skipped alloc is nan")
            return None

        return ticker, allocation, mu, sigma_squared

    except Exception as e:
        print(f"Warning: Skipping {ticker} due to calculation error: {e}")
        return None