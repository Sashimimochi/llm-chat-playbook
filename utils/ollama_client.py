"""Small HTTP client for the Ollama API.

The client deliberately uses Ollama's stable HTTP endpoints rather than local
model files, so the Streamlit application works with the Compose Ollama service
and with a local Ollama installation.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable

import requests


class OllamaClientError(RuntimeError):
    """Raised when Ollama cannot complete a request."""


class OllamaClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise OllamaClientError(
                f"Ollama ({self.base_url}) に接続できませんでした: {exc}"
            ) from exc

        if payload.get("error"):
            raise OllamaClientError(str(payload["error"]))

        return sorted(
            {
                str(model["name"])
                for model in payload.get("models", [])
                if model.get("name")
            }
        )

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": options,
            "stream": False,
        }
        if tools is not None:
            payload["tools"] = tools

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(10, 300),
            )
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise OllamaClientError(f"Ollama chat request failed: {exc}") from exc

        if result.get("error"):
            raise OllamaClientError(str(result["error"]))
        message = result.get("message")
        if not isinstance(message, dict):
            raise OllamaClientError("Ollama API response did not contain a message.")
        return message

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterable[str]:
        payload = {
            "model": model,
            "messages": messages,
            "options": options,
            "stream": True,
        }
        response = None
        collected = ""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=(10, 300),
            )
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8")
                chunk = json.loads(raw_line)
                if chunk.get("error"):
                    raise OllamaClientError(str(chunk["error"]))
                content = chunk.get("message", {}).get("content") or ""
                if content:
                    collected += content
                    yield collected
        except (requests.RequestException, ValueError) as exc:
            raise OllamaClientError(f"Ollama streaming request failed: {exc}") from exc
        finally:
            if response is not None:
                response.close()
