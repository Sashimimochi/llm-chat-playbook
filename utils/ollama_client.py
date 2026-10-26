"""Small HTTP client for the Ollama API."""
from __future__ import annotations

import json
import os
from typing import Any, Generator

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

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generator[str, None, dict[str, Any]]:
        """Stream assistant text and return its final message, including tool calls."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": options,
            "stream": True,
        }
        if tools is not None:
            payload["tools"] = tools

        response = None
        content = ""
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
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

                message = chunk.get("message", {})
                content += message.get("content") or ""
                for position, call in enumerate(message.get("tool_calls") or []):
                    index = call.get("index", position)
                    saved_call = tool_calls_by_index.setdefault(
                        index, {"function": {"arguments": {}}}
                    )
                    if call.get("id"):
                        saved_call["id"] = call["id"]
                    if call.get("type"):
                        saved_call["type"] = call["type"]

                    function = call.get("function") or {}
                    saved_function = saved_call["function"]
                    if function.get("name"):
                        saved_function["name"] = function["name"]
                    arguments = function.get("arguments")
                    if isinstance(arguments, dict):
                        saved_function["arguments"].update(arguments)

                if message.get("content"):
                    yield content
        except (requests.RequestException, ValueError) as exc:
            raise OllamaClientError(f"Ollama streaming request failed: {exc}") from exc
        finally:
            if response is not None:
                response.close()

        final_message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls_by_index:
            final_message["tool_calls"] = [
                tool_calls_by_index[index] for index in sorted(tool_calls_by_index)
            ]
        return final_message
