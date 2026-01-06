import json
import os
import requests
import streamlit as st
from datetime import datetime
from utils.mylogger import getLogger, set_logger
from llama_cpp import Llama
from utils.indexer import create_index, load_index, tokenize
set_logger()
logger = getLogger(__file__)

if "conversation" not in st.session_state:
    st.session_state.conversation = []

MODEL_DIR="./model"

MODEL_LIST = {
    "Calm2": {
        "repo_id": "TheBloke/calm2-7B-chat-GGUF",
        "filename": "calm2-7b-chat.Q5_K_M.gguf",
        "chat_format": None
    },
    "DeepSeek": {
        "repo_id": "mmnga/cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-gguf",
        "filename": "cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-Q4_K_M.gguf",
        "chat_format": None
    },
    "gpt-oss": {
        "repo_id": "unsloth/gpt-oss-20b-GGUF",
        "filename": "gpt-oss-20b-Q4_K_M.gguf",
        "chat_format": None
    },
    "gpt-oss-jp": {
        "repo_id": "Rakushaking/unsloth-gpt-oss-jp-reasoning-finetuned",
        "filename": "gpt-oss-jp-reasoning-finetuned-20250827-142614.gguf",
        "chat_format": None
    },
    "Swallow": {
        "repo_id": "mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf",
        "filename": "Llama-3.1-Swallow-8B-Instruct-v0.5-Q4_K_M.gguf",
        "chat_format": None
    },
    "FunctionCalling-ja": {
        "repo_id": "TheBloke/calm2-7B-chat-GGUF",
        "filename": "calm2-7b-chat.Q5_K_M.gguf",
        "chat_format": "chatml-function-calling"
    },
    "FunctionCalling-en": {
        "repo_id": "",
        "filename": "",
        "chat_format": "chatml-function-calling"
    },
}

class LLM:
    def __init__(
        self,
        search_type = "similarity_score_threshold",
        score_threshold = 0.8,
        window_size = 10
    ):
        self.MAX_TOKENS = 2048
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.search_num = 5
        self.window_size = window_size

    def vector_search(self, query, top_k=1):
        vector_store = load_index()
        if vector_store is None:
            return None
        keywords = tokenize(query)
        retriever = vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs={
                "score_threshold": self.score_threshold
            }
        )
        context_docs = retriever.get_relevant_documents(query, k=self.search_num)
        if len(context_docs) == 0:
            logger.warning("No documents hit.")
            return None
        logger.debug(keywords)
        for doc in context_docs:
            logger.debug(doc.metadata.get("kw"))
        res = [doc for doc in context_docs if all(x in keywords for x in doc.metadata.get("kw"))]
        logger.warning("No documents hit exactly.")
        return res[:top_k] if len(res) > 0 else context_docs[:top_k]

    def load_model(self, model_name):
        local_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name].get("filename"))
        if os.path.exists(local_path):
            return self.load_model_from_local(local_path)
        else:
            return self.load_model_from_remote(model_name)

    @st.cache_resource
    def load_model_from_local(_self, model_path):
        llm = Llama(model_path, n_ctx=_self.MAX_TOKENS)
        return llm

    @st.cache_resource
    def load_model_from_remote(_self, model_name):
        model = MODEL_LIST[model_name]
        llm = Llama.from_pretrained(
            repo_id = model["repo_id"],
            filename = model["filename"],
            verbose = False,
            n_ctx = _self.MAX_TOKENS,
            chat_format = model["chat_format"]
        )
        return llm

    def on_input_change(self, model_name, top_k=1):
        user_message = f"{st.session_state.user_message}"
        logger.info(f"user message:{user_message}")
        docs = self.vector_search(user_message, top_k=top_k)
        if docs:
            context = "\n".join([doc.page_content for doc in docs])
            urls = [doc.metadata.get("url") for doc in docs]
            if len(context) > self.MAX_TOKENS * 1.2:
                context = context[:int(self.MAX_TOKENS * 1.2)]
            template = """USER: あなたはユーザーの質問に答えるAIアシスタントBotです。
ユーザーの質問に対して適切なアドバイスを答えます。
情報として、以下の内容を参考にしてください。
====
{context}
====
さて、「{user_message}」という質問に対して、上記の情報をもとに、答えを考えてみましょう。
ASSISTANT:
""".format(user_message=user_message, context=context)
        else:
            template = """USER: {user_message} ASSISTANT: """.format(user_message=user_message)
            urls = []
        st.session_state.conversation.append(template)
        prompt = "\n".join(st.session_state.conversation[-self.window_size * 3 + 1:])
        msg = ""
        for msg in self.generate(model_name, prompt):
            yield msg
        else:
            st.session_state.urls = urls
            st.session_state.conversation.append(msg)

    def generate(self, model_name, prompt):
        if MODEL_LIST[model_name]["chat_format"] == "chatml-function-calling":
            prompt = self.generate_by_fc(model_name, prompt)
        return self.generate_chat(model_name, prompt)

    def generate_chat(self, model_name, prompt):
        llm = self.load_model(model_name)
        logger.info(f"prompt for model: {prompt}")
        streamer = llm(
            prompt,
            temperature=0.7,
            top_p=0.3,
            top_k=20,
            repeat_penalty=1.1,
            max_tokens=self.MAX_TOKENS,
            stop=["SYSTEM:", "USER:", "ASSISTANT:"],
            stream=True
        )
        partial_message = ""
        for msg in streamer:
            partial_message += msg.get("choices")[0].get("text")
            yield partial_message

    def get_location_codes(self):
        with open("data/cities.json") as f:
            location_codes = json.load(f)
        return location_codes

    def location_weather(self, location):
        location_codes = self.get_location_codes()
        city_code = location_codes.get(location)
        if city_code is None:
            return f"申し訳ありませんが、その地域の天気は取得できませんでした。取得可能な地域は以下の通りです。{list(location_codes.keys())}"
        url = f"https://weather.tsukumijima.net/api/forecast?city={city_code}"
        resp = requests.get(url).json()
        msg = f"{resp['title']}の{resp['description']['publicTimeFormatted']}の天気は{resp['description']['text']}"
        return msg

    def build_tools(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "location_weather",
                "parameters": {
                    "type": "object",
                    "title": "location_weather",
                    "properties": {
                        "location": {
                            "title": "Location",
                            "type": "string"
                        }
                    },
                    "required": [ "location" ]
                }
            }
        }]
        return tools

    def build_tool_choice(self):
        tool_choice = {
            "type": "function",
            "function": {
                "name": "location_weather"
            }
        }
        return tool_choice

    def generate_by_fc(self, model_name, prompt):
        llm = self.load_model(model_name)
        output = llm.create_chat_completion(
            messages = [
                {
                    "role": "system",
                    "content": f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions/ The assistant calls functions with appropriate input when necessary. 抽出結果は日本語にしてくください。抽出結果は可能な限り次のいずれかに寄せてください。{self.get_location_codes().keys()}"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            tools = self.build_tools(),
            tool_choice = self.build_tool_choice()
        )
        fc = output.get("choices")[0].get("message").get("function_call")
        funcname = fc.get("name")
        arguments = json.loads(fc.get("arguments"))
        logger.info(f"function calling: {fc}")
        method = getattr(self, funcname)
        info = method(**arguments)
        logger.info(f"{funcname}: {info}")

        new_prompt = f"""{prompt}わかりました。
USER: 調べたところ、以下のような情報が得られました。
================
{info}
================
上記の情報を整理してユーザにわかりやすく伝えてください。
ASSISTANT: """
        return new_prompt

def main():
    llm = LLM()
    st.title("Assistant Bot")

    if st.button("Create index"):
        with st.spinner("Creating index..."):
            create_index()

    st.write("セッションをリセットする場合はページをリロードしてください")
    top_k = st.number_input("参考情報", min_value=0, max_value=10, value="min", step=1)

    model_name = st.selectbox("Select use model", MODEL_LIST.keys())

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "私はお助けBotです。何かお手伝いできることがあれば聞いてください。",
                "timestamp": datetime.now()
            }
        ]

    # Custom CSS for timestamp styling
    st.markdown("""
    <style>
    .timestamp {
        font-size: 0.7em;
        color: #888;
        text-align: right;
        margin-top: 0.5em;
    }
    </style>
    """, unsafe_allow_html=True)

    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display timestamp
            timestamp_str = message.get("timestamp", datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
            latency_str = ""
            if message["role"] == "assistant" and "latency" in message:
                latency_str = f" (レイテンシー: {message['latency']:.2f}秒)"
            st.markdown(f'<div class="timestamp">{timestamp_str}{latency_str}</div>', unsafe_allow_html=True)

    # react to user input
    if prompt := st.chat_input("What is up?", key="user_message"):
        # Record user message timestamp
        user_timestamp = datetime.now()
        # display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            st.markdown(f'<div class="timestamp">{user_timestamp.strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
        # add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": user_timestamp
        })
        # display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Assistant thinking..."):
                placeholder = st.empty()
                timestamp_placeholder = st.empty()
                msg = "" # msgを空文字列で初期化
                for msg in llm.on_input_change(model_name=model_name, top_k=top_k):
                    placeholder.markdown(msg)
                refs = []
                for url in st.session_state.urls:
                    refs.append(f"\n- {url}")
                refs_msg = f"\n\n参考資料:{''.join(refs)}" if len(refs) > 0 else ""
                placeholder.markdown(msg+refs_msg)
                # Record assistant response timestamp and calculate latency
                assistant_timestamp = datetime.now()
                latency = (assistant_timestamp - user_timestamp).total_seconds()
                timestamp_str = assistant_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                timestamp_placeholder.markdown(f'<div class="timestamp">{timestamp_str} (レイテンシー: {latency:.2f}秒)</div>', unsafe_allow_html=True)
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": msg+refs_msg,
            "timestamp": assistant_timestamp,
            "latency": latency
        })

if __name__ == "__main__":
    main()
