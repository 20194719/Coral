"""Ordered concurrent requests to OpenAI-compatible model services."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time

from openai import OpenAI


class LLMClient:
    def __init__(self, model, endpoints, workers, output, resume=False, options=None):
        """Initialize model state."""
        self.model = model
        self.endpoints = endpoints
        if options is None:
            from src.config import deployment

            options = deployment("imdb")["llm_options"]
        self.options = dict(options)
        self.clients = [
            OpenAI(
                base_url=url,
                api_key=os.getenv(self.options["api_key_env"], "EMPTY"),
                timeout=self.options["timeout_seconds"],
                max_retries=0,
            )
            for url in endpoints
        ]
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.lock = threading.Lock()
        self.counts = [0] * len(endpoints)
        self.requests = self.failures = self.retries = 0
        journal = Path(output) / "llm-requests.jsonl"
        if resume and journal.exists():
            with journal.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    self.requests = max(self.requests, record["id"] + 1)
                    self.counts[endpoints.index(record["endpoint"])] += 1
        manifest = Path(output) / "manifest.json"
        if resume and manifest.exists():
            previous = json.loads(manifest.read_text()).get("llm", {})
            self.requests = max(self.requests, previous.get("requests", 0))
            self.failures = previous.get("failures", 0)
            self.retries = previous.get("retries", 0)
        self.log = journal.open("a" if resume else "x", encoding="utf-8")

    def _call(self, request):
        """Execute one Qwen request."""
        index, prompt, max_tokens, response_format = request
        endpoint = index % len(self.clients)
        for attempt in range(self.options["max_attempts"]):
            try:
                response = self.clients[endpoint].chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=self.options["temperature"],
                    top_p=self.options["top_p"],
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": self.options["enable_thinking"]
                        }
                    },
                    **({"response_format": response_format} if response_format else {}),
                )
                choice = response.choices[0]
                reply = (choice.message.content or "").strip()
                if not reply:
                    raise ValueError("Model returned an empty response")
                with self.lock:
                    self.counts[endpoint] += 1
                    self.log.write(
                        json.dumps(
                            dict(
                                id=index,
                                endpoint=self.endpoints[endpoint],
                                max_tokens=max_tokens,
                                prompt=prompt,
                                reply=reply,
                                finish_reason=choice.finish_reason,
                                response_format=response_format,
                                usage=response.usage.model_dump()
                                if response.usage
                                else None,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    self.log.flush()
                    total = sum(self.counts)
                    if total % 500 == 0:
                        print(f"[llm] {total} completed requests", flush=True)
                return reply
            except Exception as exc:
                if attempt + 1 == self.options["max_attempts"] or getattr(
                    exc, "status_code", None
                ) in (
                    400,
                    401,
                    403,
                    404,
                    413,
                ):
                    with self.lock:
                        self.failures += 1
                    raise RuntimeError(
                        f"LLM request {index} failed at {self.endpoints[endpoint]}: {exc}"
                    ) from exc
                with self.lock:
                    self.retries += 1
                time.sleep(0.5 * 2**attempt)

    def complete(self, prompts, max_tokens, response_format=None):
        """Submit prompts to the Qwen service."""
        with self.lock:
            start = self.requests
            self.requests += len(prompts)
        return list(
            self.pool.map(
                self._call,
                (
                    (start + i, p, max_tokens, response_format)
                    for i, p in enumerate(prompts)
                ),
            )
        )

    def statistics(self):
        """Return request statistics."""
        return dict(
            requests=self.requests,
            completed=sum(self.counts),
            failures=self.failures,
            retries=self.retries,
            per_endpoint=dict(zip(self.endpoints, self.counts)),
        )

    def close(self):
        """Close inference resources."""
        self.pool.shutdown(wait=True)
        for client in self.clients:
            client.close()
        self.log.close()
