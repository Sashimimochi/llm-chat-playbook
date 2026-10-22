import json
import logging
import os
import re
from datetime import datetime
from typing import Any

import requests
import streamlit as st

from utils.indexer import create_index, load_index, tokenize
from utils.mylogger import getLogger, set_logger
from utils.ollama_client import OllamaClient, OllamaClientError

set_logger()
logger = getLogger(__file__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MAX_CONTEXT_TOKENS = 4096
DEFAULT_SYSTEM_PROMPT = (
    "あなたはユーザーの質問に答える日本語のAIアシスタントです。"
    "正確で分かりやすい回答をしてください。"
)


def remove_thinking_tags(text: str) -> str:
    """Remove complete reasoning tags from model output when requested."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


class LLM:
    def __init__(
        self,
        search_type: str = "similarity_score_threshold",
        score_threshold: float = 0.8,
        window_size: int = 10,
    ) -> None:
        self.client = OllamaClient(OLLAMA_BASE_URL)
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.search_num = 5
        self.window_size = window_size

    def vector_search(self, query: str, top_k: int = 1):
        vector_store = load_index()
        if vector_store is None:
            return None

        keywords = tokenize(query)
        retriever = vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs={"score_threshold": self.score_threshold},
        )
        context_docs = retriever.get_relevant_documents(query, k=self.search_num)
        if not context_docs:
            logger.warning("No documents hit.")
            return None

        logger.debug(keywords)
        for doc in context_docs:
            logger.debug(doc.metadata.get("kw"))
        exact_matches = [
            doc
            for doc in context_docs
            if all(keyword in keywords for keyword in doc.metadata.get("kw", []))
        ]
        return exact_matches[:top_k] if exact_matches else context_docs[:top_k]

    def get_location_codes(self) -> dict[str, str]:
        with open("data/cities.json", encoding="utf-8") as city_file:
            return json.load(city_file)

    def location_weather(self, location: str) -> str:
        city_code = self.get_location_codes().get(location)
        if city_code is None:
            cities = "、".join(self.get_location_codes().keys())
            return (
                f"申し訳ありませんが「{location}」の天気は取得できませんでした。"
                f"取得可能な地域は以下の通りです: {cities}"
            )

        url = f"https://weather.tsukumijima.net/api/forecast?city={city_code}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        forecast = response.json()
        title = forecast.get("title", location)
        description = forecast.get("description", {}).get("text", "天気情報がありません")
        published = forecast.get("description", {}).get("publicTimeFormatted", "")
        return f"{title}の{published}の天気は{description}"

    def build_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "location_weather",
                    "description": "日本の指定された地域の天気予報を取得します。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "天気を調べたい地域名。例: 東京、横浜、大阪",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

    def build_options(
        self,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "num_predict": max_tokens or 2048,
            "num_ctx": MAX_CONTEXT_TOKENS,
        }

    def build_messages(self, user_message: str, context: str | None = None):
        system_content = DEFAULT_SYSTEM_PROMPT
        if context:
            system_content += (
                "\n\n回答の参考情報は以下です。参考情報にない内容は推測せず、"
                "必要に応じて情報がないことを説明してください。\n"
                "====\n"
                f"{context}\n"
                "===="
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]
        # The last entry is the current user message, already added by main().
        history = st.session_state.messages[:-1]
        history = [
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        if self.window_size > 0:
            history = history[-self.window_size * 2 :]
        else:
            history = []
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def run_tool(self, name: str, arguments: Any) -> str:
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if name != "location_weather":
            raise ValueError(f"Unsupported tool: {name}")
        if not isinstance(arguments, dict) or not isinstance(
            arguments.get("location"), str
        ):
            raise ValueError("location_weather requires a string location.")
        return self.location_weather(arguments["location"])

    def on_input_change(
        self,
        model_name: str,
        top_k: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.3,
        top_k_param: int = 20,
        repeat_penalty: float = 1.1,
        max_tokens: int | None = None,
        use_function_calling: bool = False,
        hide_thinking: bool = False,
    ):
        user_message = str(st.session_state.user_message)
        logger.info("user message: %s", user_message)

        docs = self.vector_search(user_message, top_k=top_k) if top_k > 0 else None
        context = "\n".join(doc.page_content for doc in docs) if docs else None
        urls = [doc.metadata.get("url") for doc in docs] if docs else []
        if context and len(context) > MAX_CONTEXT_TOKENS * 1.2:
            context = context[: int(MAX_CONTEXT_TOKENS * 1.2)]

        messages = self.build_messages(user_message, context)
        options = self.build_options(
            temperature, top_p, top_k_param, repeat_penalty, max_tokens
        )
        logger.info("Ollama model: %s", model_name)

        try:
            if use_function_calling:
                assistant_message = self.client.chat(
                    model=model_name,
                    messages=messages,
                    options=options,
                    tools=self.build_tools(),
                )
                tool_calls = assistant_message.get("tool_calls") or []
                if not tool_calls:
                    text = assistant_message.get("content") or ""
                    yield remove_thinking_tags(text) if hide_thinking else text
                else:
                    messages.append(assistant_message)
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        tool_name = function.get("name", "")
                        arguments = function.get("arguments", {})
                        try:
                            result = self.run_tool(tool_name, arguments)
                        except (ValueError, json.JSONDecodeError) as exc:
                            result = f"ツールを実行できませんでした: {exc}"
                        messages.append({"role": "tool", "content": result})
                    for partial in self.client.stream_chat(
                        model=model_name, messages=messages, options=options
                    ):
                        yield remove_thinking_tags(partial) if hide_thinking else partial
            else:
                for partial in self.client.stream_chat(
                    model=model_name, messages=messages, options=options
                ):
                    yield remove_thinking_tags(partial) if hide_thinking else partial
        except (OllamaClientError, requests.RequestException, ValueError) as exc:
            logger.exception("Ollama response generation failed")
            yield f"Ollamaへのリクエストに失敗しました: {exc}"
        finally:
            st.session_state.urls = urls


def main() -> None:
    llm = LLM()
    st.title("Assistant Bot")

    try:
        model_names = llm.client.list_models()
    except OllamaClientError as exc:
        st.error(
            f"Ollamaに接続できません。OLLAMA_BASE_URL={OLLAMA_BASE_URL} を確認してください。"
        )
        st.code(str(exc))
        st.stop()

    if not model_names:
        st.error(
            "Ollamaにモデルがありません。.env の OLLAMA_INIT_MODELS を設定して "
            "make launch を実行するか、ollama pull でモデルを取得してください。"
        )
        st.stop()

    with st.sidebar:
        st.header("モデルパラメータ設定")
        temperature = st.slider(
            "Temperature（温度）",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="値が高いほど出力がランダムになります。",
        )
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="nucleus samplingのパラメータです。",
        )
        top_k_param = st.slider(
            "Top K",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            help="各ステップで考慮する上位Kトークンの数です。",
        )
        repeat_penalty = st.slider(
            "Repeat Penalty（繰り返しペナルティ）",
            min_value=1.0,
            max_value=2.0,
            value=1.1,
            step=0.05,
            help="値を大きくすると繰り返しを抑制します。",
        )
        max_tokens = st.slider(
            "Max Tokens（最大生成トークン数）",
            min_value=256,
            max_value=4096,
            value=2048,
            step=256,
        )

    if st.button("Create index"):
        with st.spinner("Creating index..."):
            create_index()

    st.write("セッションをリセットする場合はページをリロードしてください")
    top_k = st.number_input("参考情報", min_value=0, max_value=10, value=1, step=1)
    model_name = st.selectbox("Select use model", model_names)
    use_function_calling = st.checkbox(
        "Function Callingで天気APIを利用する", value=False
    )
    hide_thinking = st.checkbox("思考タグ（think / thinking）を非表示にする", value=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "私はお助けBotです。何かお手伝いできることがあれば聞いてください。",
                "timestamp": datetime.now(),
            }
        ]
    if "urls" not in st.session_state:
        st.session_state.urls = []

    st.markdown(
        """
        <style>
        .timestamp {
            font-size: 0.7em;
            color: #888;
            text-align: right;
            margin-top: 0.5em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            timestamp = message.get("timestamp", datetime.now())
            if not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            latency = message.get("latency")
            latency_text = f" (レイテンシー: {latency:.2f}秒)" if latency is not None else ""
            st.markdown(
                f'<div class="timestamp">{timestamp:%Y-%m-%d %H:%M:%S}{latency_text}</div>',
                unsafe_allow_html=True,
            )

    if prompt := st.chat_input("What is up?", key="user_message"):
        user_timestamp = datetime.now()
        with st.chat_message("user"):
            st.markdown(prompt)
            st.markdown(
                f'<div class="timestamp">{user_timestamp:%Y-%m-%d %H:%M:%S}</div>',
                unsafe_allow_html=True,
            )
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "timestamp": user_timestamp}
        )

        with st.chat_message("assistant"):
            with st.spinner("Assistant thinking..."):
                placeholder = st.empty()
                timestamp_placeholder = st.empty()
                message_text = ""
                for message_text in llm.on_input_change(
                    model_name=model_name,
                    top_k=top_k,
                    temperature=temperature,
                    top_p=top_p,
                    top_k_param=top_k_param,
                    repeat_penalty=repeat_penalty,
                    max_tokens=max_tokens,
                    use_function_calling=use_function_calling,
                    hide_thinking=hide_thinking,
                ):
                    placeholder.markdown(message_text)
                reference_urls = [
                    f"\n- {url}" for url in st.session_state.get("urls", []) if url
                ]
                references = (
                    f"\n\n参考資料:{''.join(reference_urls)}" if reference_urls else ""
                )
                placeholder.markdown(message_text + references)
                assistant_timestamp = datetime.now()
                latency = (assistant_timestamp - user_timestamp).total_seconds()
                timestamp_placeholder.markdown(
                    f'<div class="timestamp">{assistant_timestamp:%Y-%m-%d %H:%M:%S} '
                    f'(レイテンシー: {latency:.2f}秒)</div>',
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": message_text + references,
                "timestamp": assistant_timestamp,
                "latency": latency,
            }
        )


if __name__ == "__main__":
    main()
