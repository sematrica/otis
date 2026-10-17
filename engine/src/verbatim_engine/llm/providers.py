"""Language-model providers for optional transcript review.

Two providers behind one interface:

* :class:`OllamaProvider` - a local Ollama server (``/api/chat``), the
  default: nothing leaves the computer;
* :class:`OpenAIProvider` - the OpenAI Chat Completions API (or any server
  that implements it, given a base URL). The API key is passed in by the app,
  which keeps it in the macOS Keychain; the engine never stores or logs it.

An endpoint that is not on this computer is *remote*: the review refuses to
send anything to it unless the request explicitly allows it (the app asks
the user each time). Only transcript text is ever sent, never audio.
"""

from __future__ import annotations

import abc
import ipaddress
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from ..errors import EngineError, InvalidParams

OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
REQUEST_TIMEOUT_S = 300.0


class LLMError(EngineError):
    def __init__(self, message: str, data: dict[str, Any] | None = None):
        super().__init__(message, code="llm_error", data=data)


def is_local(url: str) -> bool:
    """True if the endpoint is on this computer (loopback)."""
    host = (urlparse(url).hostname or "").strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validated_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise InvalidParams("The provider URL must be an http or https address.", data={"url": url})
    if parsed.scheme == "http" and not is_local(url):
        raise InvalidParams("Remote providers must use https.", data={"url": url})
    return url.rstrip("/")


@dataclass
class Exchange:
    """One request and its response, kept for traceability."""

    request: dict[str, Any]
    response: str | None
    seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"request": self.request, "response": self.response, "seconds": round(self.seconds, 2),
                "error": self.error}


class Provider(abc.ABC):
    name: str
    display_name: str

    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        self.base_url = _validated_url(base_url)
        self.model = model
        self._api_key = api_key

    @property
    def remote(self) -> bool:
        return not is_local(self.base_url)

    def _client(self):
        import httpx2

        # loopback servers must not be reached through a proxy
        return httpx2.Client(timeout=httpx2.Timeout(30.0, read=REQUEST_TIMEOUT_S), trust_env=self.remote,
                             follow_redirects=False)

    @abc.abstractmethod
    def list_models(self) -> list[str]: ...

    @abc.abstractmethod
    def _complete(self, client, system: str, user: str, schema: dict[str, Any]) -> str: ...

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> tuple[dict[str, Any], Exchange]:
        """Ask for a JSON answer that follows ``schema``; returns it parsed and
        the exchange record. Raises :class:`LLMError` (with the exchange in
        ``data``) on failure."""
        request = {"provider": self.name, "model": self.model, "endpoint": self.base_url, "system": system,
                   "user": user}
        started = time.monotonic()
        content = None
        try:
            with self._client() as client:
                content = self._complete(client, system, user, schema)
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("the answer is not a JSON object")
        except LLMError as exc:
            exchange = Exchange(request, content, time.monotonic() - started, exc.message)
            raise LLMError(exc.message, data={"exchange": exchange.to_dict()}) from exc
        except ValueError as exc:
            exchange = Exchange(request, content, time.monotonic() - started, f"unreadable answer: {exc}")
            raise LLMError("The model's answer was not valid JSON.", data={"exchange": exchange.to_dict()}) from exc
        except Exception as exc:  # noqa: BLE001 - network errors of the HTTP client
            exchange = Exchange(request, content, time.monotonic() - started, f"{type(exc).__name__}: {exc}")
            raise LLMError(f"The {self.display_name} request failed: {type(exc).__name__}.",
                           data={"exchange": exchange.to_dict()}) from exc
        return parsed, Exchange(request, content, time.monotonic() - started)

    @staticmethod
    def _check(response, what: str) -> dict[str, Any]:
        if response.status_code >= 400:
            detail = response.text[:500]
            raise LLMError(f"{what} answered HTTP {response.status_code}.", data={"detail": detail})
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError(f"{what} sent an unreadable response.") from exc


class OllamaProvider(Provider):
    name = "ollama"
    display_name = "Ollama"

    def __init__(self, model: str, base_url: str | None = None):
        super().__init__(base_url or OLLAMA_DEFAULT_URL, model)

    def list_models(self) -> list[str]:
        with self._client() as client:
            data = self._check(client.get(f"{self.base_url}/api/tags"), "Ollama")
        return sorted(m["name"] for m in data.get("models", []) if m.get("name"))

    def _complete(self, client, system: str, user: str, schema: dict[str, Any]) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,  # structured output (Ollama >= 0.5)
            "options": {"temperature": 0, "seed": 0},
        }
        data = self._check(client.post(f"{self.base_url}/api/chat", json=body), "Ollama")
        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise LLMError("Ollama sent no answer.")
        return content


class OpenAIProvider(Provider):
    name = "openai"
    display_name = "OpenAI"

    def __init__(self, model: str, api_key: str | None, base_url: str | None = None):
        super().__init__(base_url or OPENAI_DEFAULT_URL, model, api_key)
        if self.remote and not api_key:
            raise InvalidParams("An OpenAI API key is required (Settings › AI Verification).")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def list_models(self) -> list[str]:
        with self._client() as client:
            data = self._check(client.get(f"{self.base_url}/models", headers=self._headers()), "OpenAI")
        return sorted(m["id"] for m in data.get("data", []) if m.get("id"))

    def _complete(self, client, system: str, user: str, schema: dict[str, Any]) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        formats = [
            {"type": "json_schema", "json_schema": {"name": "transcript_review", "schema": schema, "strict": True}},
            {"type": "json_object"},  # servers without structured outputs (the prompt describes the format)
        ]
        last: LLMError | None = None
        for response_format in formats:
            for temperature in (0, None):  # some models only accept their default temperature
                body: dict[str, Any] = {"model": self.model, "messages": messages, "response_format": response_format}
                if temperature is not None:
                    body["temperature"] = temperature
                response = client.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers())
                if response.status_code == 400:
                    last = LLMError("OpenAI rejected the request.", data={"detail": response.text[:500]})
                    continue
                data = self._check(response, "OpenAI")
                choices = data.get("choices") or []
                content = (choices[0].get("message") or {}).get("content") if choices else None
                if not isinstance(content, str):
                    raise LLMError("OpenAI sent no answer.")
                return content
        raise last or LLMError("OpenAI rejected the request.")


def make_provider(name: str, model: str | None, *, base_url: str | None = None, api_key: str | None = None,
                  require_model: bool = True) -> Provider:
    if require_model and (not model or not isinstance(model, str)):
        raise InvalidParams("Choose a model for AI verification.")
    model = model or ""
    if name == "ollama":
        return OllamaProvider(model, base_url)
    if name == "openai":
        return OpenAIProvider(model, api_key, base_url)
    raise InvalidParams(f"Unknown AI provider '{name}'.", data={"providers": ["ollama", "openai"]})
