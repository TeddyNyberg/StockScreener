

from locust import HttpUser, task, between
import random

class CacheStressUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def test_popular_ticker(self):
        self.client.get("/chart?tickers=AAPL")

    @task(1)
    def test_random_tickers(self):
        random_ticker = f"TICK{random.randint(1, 1000)}"
        self.client.get(f"/chart?tickers={random_ticker}")
