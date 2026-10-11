"""Standalone multi-turn chat CLI for Ollama.

Usage:
  python ollama_chat_app.py --model qwen3:8b --window-size 10
"""

from __future__ import annotations

import argparse
from typing import Any

import ollama

DEFAULT_SYSTEM_PROMPT = "あなたは親切なアシスタントです。日本語で回答してください。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OllamaマルチターンチャットCLI")
    parser.add_argument("--model", default="qwen3:8b", help="利用するOllamaモデル名")
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="文脈として渡す直近ターン数（systemメッセージを除く）",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=2048,
        help="1回の応答で生成する最大トークン数",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="初期システムプロンプト",
    )
    return parser


def build_context(conversation: list[dict[str, str]], window_size: int) -> list[dict[str, str]]:
    system_message = conversation[:1]
    recent_messages = conversation[-(window_size * 2) :] if window_size > 0 else []
    return system_message + recent_messages


def stream_chat(
    conversation: list[dict[str, str]],
    *,
    model: str,
    window_size: int,
    num_predict: int,
) -> str:
    context = build_context(conversation, window_size)
    response = ollama.chat(
        model=model,
        messages=context,
        stream=True,
        options={"num_predict": num_predict},
    )

    partial_message = ""
    print("アシスタント: ", end="", flush=True)
    for chunk in response:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        partial_message += token
    print()
    return partial_message


def run_chat(model: str, window_size: int, num_predict: int, system_prompt: str) -> None:
    conversation: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print("Ollamaマルチターンチャットを開始します。終了するには /exit を入力してください。")
    print(f"model={model}, window_size={window_size}, num_predict={num_predict}")

    while True:
        user_message = input("あなた: ").strip()
        if not user_message:
            continue
        if user_message in {"/exit", "exit", "quit", "/quit"}:
            print("チャットを終了します。")
            break

        conversation.append({"role": "user", "content": user_message})
        assistant_message = stream_chat(
            conversation,
            model=model,
            window_size=window_size,
            num_predict=num_predict,
        )
        conversation.append({"role": "assistant", "content": assistant_message})


def main() -> None:
    args = build_parser().parse_args()
    run_chat(
        model=args.model,
        window_size=args.window_size,
        num_predict=args.num_predict,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
