"""Provider-neutral records. HTTP is standard-library only; no SDK retries."""

import asyncio
import json
import os
import socket
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class ProviderError(Exception):
    def __init__(self, message, *, retryable=False, status=None, delivery_unknown=False, retry_after=None, code=None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.delivery_unknown = delivery_unknown
        self.retry_after = retry_after
        self.code = code


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def retry_delay(value):
    try:
        return max(0, float(value))
    except (ValueError, TypeError):
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


@dataclass
class ModelResponse:
    text: str
    model_id: str
    finish_reason: str | None = None
    generation_limit_reached: bool = False
    usage: dict = field(default_factory=dict)
    request_id: str | None = None
    provider_response: dict = field(default_factory=dict)
    history_message: dict | None = None


class Adapter:
    def __init__(self, config, timeout=120):
        self.config = config
        self.timeout = timeout

    def prepare(self, messages):
        raise NotImplementedError

    def parse(self, data, headers):
        raise NotImplementedError

    def headers(self):
        key = os.environ.get(self.config.get("api_key_env", ""), "")
        return {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}

    async def generate(self, messages):
        endpoint, payload = self.prepare(messages)
        return await asyncio.to_thread(self._request, endpoint, payload)

    def _request(self, endpoint, payload):
        request = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=self.headers(), method="POST")
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
                return self.parse(data, response.headers)
        except urllib.error.HTTPError as exc:
            # Inspect billing signals without logging provider bodies or secrets.
            try:
                error = json.loads(exc.read(65536)).get("error", {})
                fields = " ".join(str(error.get(k, "")) for k in ("code", "type", "message")).lower()
            except (ValueError, AttributeError, TypeError):
                fields = ""
            billing = any(term in fields for term in ("insufficient_quota", "billing", "credit balance", "spend limit", "quota exceeded", "quota_exceeded", "daily limit"))
            delay = retry_delay(exc.headers.get("Retry-After")) if exc.headers else None
            retryable = exc.code in {429, 503, 529} and not billing
            unknown = exc.code in {408, 500, 502, 504}
            code = "billing_or_quota" if billing else "temporary_rate_or_overload" if retryable else "http_error"
            raise ProviderError(f"HTTP {exc.code} ({code}); check access, balance and model settings",
                                retryable=retryable, status=exc.code, delivery_unknown=unknown,
                                retry_after=delay, code=code) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError):
            raise ProviderError("Network request failed; delivery may be unknown. Automatic retry disabled.",
                                delivery_unknown=True) from None
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError("Unexpected provider response; delivery may be unknown. Inspect API/model compatibility.",
                                delivery_unknown=True) from None


class MockAdapter(Adapter):
    def prepare(self, messages):
        return "mock://offline", {"messages": messages}

    async def generate(self, messages):
        index = sum(message["role"] == "user" for message in messages) - 1
        replies = self.config["responses"]
        return ModelResponse(replies[index % len(replies)], self.config["model"], "mock_stop", provider_response={"synthetic": True})
