import streamlit as st
from utils.mylogger import getLogger, set_logger
from llama_cpp import Llama
from utils.indexer import create_index, load_index, tokenize
set_logger()
logger = getLogger(__file__)

MAX_TOKENS = 2048

def vector_search(query, top_k=1):
	vector_store = load_index()
	keywords = tokenize(query)
	retriever = vector_store.as_retriever(
		search_type="similarity_score_threshold",
		search_kwargs={
			"score_threshold": 1.0
		}
	)
	context_docs = retriever.get_relevant_documents(query, k=5)
	if len(context_docs) == 0:
		logger.warning("No documents hit.")
		return None
	logger.debug(keywords)
	for doc in context_docs:
		logger.debug(doc.metadata.get("kw"))
	res = [doc for doc in context_docs if all(x in keywords for x in doc.metadata.get("kw"))]
	logger.warning("No documents hit exactly.")
	return res[:top_k] if len(res) > 0 else context_docs[:top_k]

@st.cache_resource
def load_model():
	model_path = "./model/calm2-7b-chat.Q5_K_M.gguf"
	llm = Llama(model_path, n_ctx=MAX_TOKENS)
	return llm

if "conversation" not in st.session_state:
	st.session_state.conversation = []

def on_input_change(window_size=3, top_k=1):
	user_message = f"{st.session_state.user_message}"
	docs = vector_search(user_message, top_k=top_k)
	if docs:
		context = "\n".join([doc.page_content for doc in docs])
		urls = [doc.metadata.get("url") for doc in docs]
		if len(context) > MAX_TOKENS * 1.2:
			context = context[:int(MAX_TOKENS * 1.2)]
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
		urls = ["なし"]
	st.session_state.conversation.append(template)
	prompt = "\n".join(st.session_state.conversation[-window_size * 3 + 1:])
	logger.info(f"prompt: {prompt}")
	for msg in generate(prompt):
		yield msg
	else:
		st.session_state.urls = urls
		st.session_state.conversation.append(msg)

def generate(prompt):
	llm = load_model()
	streamer = llm(
		prompt,
		temperature=0.7,
		top_p=0.3,
		top_k=20,
		repeat_penalty=1.1,
		max_tokens=MAX_TOKENS,
		stop=["SYSTEM:", "USER:", "ASSISTANT:"],
		stream=True
	)
	partial_message = ""
	for msg in streamer:
		partial_message += msg.get("choices")[0].get("text")
		yield partial_message

st.title("Assistant Bot")

if st.button("Create index"):
	with st.spinner("Creating index..."):
		create_index()

st.write("セッションをリセットする場合はページをリロードしてください")
top_k = st.number_input("参考情報", min_value=1, max_value=10, value="min", step=1)

# initialize chat history
if "messages" not in st.session_state:
	st.session_state.messages = [
		{
			"role": "assistant",
			"content": "私はお助けBotです。何かお手伝いできることがあれば聞いてください。"
		}
	]

# display chat messages from history on app rerun
for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])

# react to user input
if prompt := st.chat_input("What is up?", key="user_message"):
	# display user message in chat message container
	st.chat_message("user").markdown(prompt)
	# add user message to chat history
	st.session_state.messages.append({"role": "user", "content": prompt})
	# display assistant response in chat message container
	with st.chat_message("assistant"):
		with st.spinner("Asisstant thinking..."):
			placeholder = st.empty()
			for msg in on_input_change(top_k=top_k):
				placeholder.markdown(msg)
			refs = []
			for url in st.session_state.urls:
				refs.append(f"\n- {url}")
			refs_msg = f"\n\n参考資料:{''.join(refs)}"
			placeholder.markdown(msg+refs_msg)
	# Add assistant response to chat history
	st.session_state.messages.append({"role": "assistant", "content": msg+refs_msg})
