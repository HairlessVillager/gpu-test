from random import randint
from locust import HttpUser, task, between


class QuickstartUser(HttpUser):
    @task
    def hello_world(self):
        start = randint(0, 9000000)
        end = start + randint(100, 800)
        sub_text = self.text[start:end]
        self.client.post(
            "/predit/text",
            json={"document": sub_text, "version": "no-version", "multilingual": False},
        )

    def on_start(self):
        with open("data/Shakespeare.txt") as f:
            self.text = f.read()
