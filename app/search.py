import yfinance as yf

def lookup_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or "shortName" not in info:
            return None

        return {
            "ticker":ticker,
            "name":info.get("shortName", "N/A"),
            "price":info.get("currentPrice", "N/A"),
            "currency":info.get("currency", "USD")
        }

    except Exception as e:
        print(f"Error looking up {ticker}: {e}")
        return None

