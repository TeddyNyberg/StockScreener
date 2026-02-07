import threading
import time
import random
from backend.app.data.data_cache import get_yfdata_cache
import requests
from scripts.utils import extract_tickers_from_response

NUM_FRONTENDS = 50
REQUESTS_PER_FRONTEND = 20
ALL_TICKERS_URL = "https://stockanalysis.com/stocks/"


def frontend_simulator(id, ticker_pool):

    for i in range(REQUESTS_PER_FRONTEND):
        # Pick a random ticker to simulate user browsing
        ticker = random.choice(ticker_pool)
        print(f"Frontend {id}: Requesting {ticker}...")

        # This calls your actual cache logic
        get_yfdata_cache([ticker], time="1Y")

        time.sleep(1)


if __name__ == "__main__":

    r = requests.get(ALL_TICKERS_URL)
    new_tickers = extract_tickers_from_response(r)
    processed_tickers = [t.replace(".", "-") for t in new_tickers]



    threads = []
    print(f"--- Starting Stress Test with {NUM_FRONTENDS} Frontends ---")

    for i in range(NUM_FRONTENDS):
        t = threading.Thread(target=frontend_simulator, args=(i,processed_tickers))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("--- Test Complete ---")