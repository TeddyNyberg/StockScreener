
from locust import HttpUser, task, between, SequentialTaskSet
import random

class CacheStressUser(HttpUser):
    wait_time = between(0.1, 0.5)

    #@task(5)
    #def test_popular_ticker(self):
    #    self.client.get("/chart?tickers=AAPL&time=1Y")

    @task(1)
    def test_random_tickers(self):
        random_ticker = f"TICK{50}"
        self.client.get(f"/chart?tickers={random_ticker}&time=1Y")

    #f"TICK{random.randint(1, 1000)}"


class CacheSequence(SequentialTaskSet):
    @task
    def first_call(self):
        self.client.get("/chart?tickers=TICK9&time=1Y")

    @task
    def second_call(self):
        self.client.get("/chart?tickers=TICK9&time=2Y")

class CacheUser(HttpUser):
    tasks = [CacheSequence]
    wait_time = between(1, 2)


