from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from utils.ollama_client import OllamaClient, OllamaClientError


def consume_stream(generator):
    chunks = []
    while True:
        try:
            chunks.append(next(generator))
        except StopIteration as stop:
            return chunks, stop.value


class OllamaClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OllamaClient("http://ollama:11434")

    @patch("utils.ollama_client.requests.get")
    def test_list_models_returns_sorted_names(self, mocked_get: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "models": [{"name": "qwen3:4b"}, {"name": "gemma4:e2b"}]
        }
        mocked_get.return_value = response

        self.assertEqual(
            self.client.list_models(),
            ["gemma4:e2b", "qwen3:4b"],
        )
        mocked_get.assert_called_once_with(
            "http://ollama:11434/api/tags", timeout=5
        )

    @patch("utils.ollama_client.requests.post")
    def test_stream_chat_passes_tools_and_returns_message(
        self, mocked_post: Mock
    ) -> None:
        response = Mock()
        response.iter_lines.return_value = [
            json.dumps({"message": {"role": "assistant", "content": "こんにちは"}}).encode(),
            json.dumps({"message": {"role": "assistant", "content": "！"}}).encode(),
        ]
        mocked_post.return_value = response
        tools = [{"type": "function", "function": {"name": "location_weather"}}]
        messages = [{"role": "user", "content": "こんにちは"}]

        chunks, result = consume_stream(
            self.client.stream_chat(
                model="gemma4:e2b",
                messages=messages,
                options={"temperature": 0.7},
                tools=tools,
            )
        )

        self.assertEqual(chunks, ["こんにちは", "こんにちは！"])
        self.assertEqual(
            result,
            {"role": "assistant", "content": "こんにちは！"},
        )
        kwargs = mocked_post.call_args.kwargs
        self.assertEqual(kwargs["json"]["messages"], messages)
        self.assertEqual(kwargs["json"]["tools"], tools)
        self.assertTrue(kwargs["json"]["stream"])
        response.close.assert_called_once()

    @patch("utils.ollama_client.requests.post")
    def test_stream_chat_collects_multiple_tool_calls(self, mocked_post: Mock) -> None:
        first_call = {
            "index": 0,
            "function": {
                "name": "location_weather",
                "arguments": {"location": "東京"},
            },
        }
        second_call = {
            "index": 1,
            "function": {
                "name": "location_weather",
                "arguments": {"location": "横浜"},
            },
        }
        response = Mock()
        response.iter_lines.return_value = [
            json.dumps(
                {"message": {"role": "assistant", "content": "", "tool_calls": [first_call]}}
            ).encode(),
            json.dumps(
                {"message": {"role": "assistant", "content": "", "tool_calls": [second_call]}}
            ).encode(),
        ]
        mocked_post.return_value = response

        chunks, result = consume_stream(
            self.client.stream_chat(
                model="gemma4:e2b",
                messages=[{"role": "user", "content": "東京と横浜の天気"}],
                options={},
                tools=[{"type": "function"}],
            )
        )

        self.assertEqual(chunks, [])
        self.assertEqual(
            result["tool_calls"],
            [first_call, second_call],
        )
        response.close.assert_called_once()

    @patch("utils.ollama_client.requests.get")
    def test_list_models_wraps_connection_errors(self, mocked_get: Mock) -> None:
        import requests

        mocked_get.side_effect = requests.ConnectionError("unavailable")
        with self.assertRaises(OllamaClientError):
            self.client.list_models()


if __name__ == "__main__":
    unittest.main()
