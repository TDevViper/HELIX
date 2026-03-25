from __future__ import annotations

"""
Locust load test for Helix.
Run with: locust -f helix/tests/load/benchmark.py --host http://localhost:8000
"""

import json  # noqa: E402
import random  # noqa: E402

try:
    from locust import HttpUser, between, task
except ImportError:
    raise SystemExit("Install locust: pip install locust")

MODELS = ["llama3", "mistral", "phi3"]
PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a haiku about distributed systems.",
    "What is the capital of France?",
    "Summarize the plot of Hamlet in one paragraph.",
]


class HelixUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def chat_stream(self) -> None:
        payload = {
            "model": random.choice(MODELS),
            "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
            "stream": True,
        }
        with self.client.post(
            "/v1/chat/completions",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def health_check(self) -> None:
        self.client.get("/v1/health")

    @task(1)
    def stats(self) -> None:
        self.client.get("/v1/stats")
