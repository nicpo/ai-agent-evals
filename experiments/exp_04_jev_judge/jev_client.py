"""Minimal HTTP client for the TypeSafe-compatible System One API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import load_env


class JevClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class JevResponse:
    payload: dict
    status: int
    request_id: str | None
    latency_ms: float


class JevClient:
    def __init__(self, base_url: str, api_key_env: str = "TYPESAFE_API_KEY",
                 timeout_seconds: float = 60):
        load_env()
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds

    def _api_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if not key:
            raise JevClientError(
                f"Missing {self.api_key_env}. Add it to the repository .env file or process environment."
            )
        return key

    def require_key(self) -> None:
        """Fail fast (before any agent run) if the API key is missing."""
        self._api_key()

    def _request(self, method: str, path: str, payload: dict | None = None) -> JevResponse:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}", data=body, method=method,
            headers={"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"},
        )
        started = perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                return JevResponse(
                    payload=json.loads(raw), status=response.status,
                    request_id=response.headers.get("x-typesafe-request-id"),
                    latency_ms=(perf_counter() - started) * 1000,
                )
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise JevClientError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except URLError as exc:
            raise JevClientError(f"Connection error calling {path}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise JevClientError(f"Invalid JSON from {path}: {exc}") from exc

    def list_models(self) -> JevResponse:
        return self._request("GET", "/v1/models")

    def system_one(self, *, model: str, state, questions: dict) -> JevResponse:
        return self._request("POST", "/v1/systemone", {
            "model": model, "state": state, "questions": questions,
        })


def validate_choice_response(response: JevResponse, question_name: str,
                             expected_options: set[str]) -> dict:
    try:
        model = response.payload["model"]
        usage = response.payload["usage"]
        answer = response.payload["answers"][question_name]
        choice = answer["choice"]
        confidence = float(answer["confidence"])
        probabilities = {str(key): float(value) for key, value in answer["probabilities"].items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise JevClientError(f"Malformed choice response: {response.payload}") from exc
    if answer.get("type") != "choice" or set(probabilities) != expected_options or choice not in expected_options:
        raise JevClientError(f"Unexpected choice schema: {answer}")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
        raise JevClientError(f"Invalid probabilities: {probabilities}")
    if abs(sum(probabilities.values()) - 1) > 0.02:
        raise JevClientError(f"Probabilities do not sum to approximately one: {probabilities}")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise JevClientError(f"Invalid confidence: {confidence}")
    return {
        "model": model, "usage": usage, "answer": answer,
        "choice": choice, "confidence": confidence, "probabilities": probabilities,
        "latency_ms": response.latency_ms, "request_id": response.request_id,
        "http_status": response.status, "raw_response": response.payload,
    }

