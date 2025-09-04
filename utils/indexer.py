import glob
import json
import os
import re

import streamlit as st
from stqdm import stqdm

import tika
tika.initVM()
from tika import parser

from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter

from hojichar.core.filter_interface import Filter
from hojichar import Compose

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from utils.mylogger import getLogger, set_logger

set_logger()
logger = getLogger(__file__)

class URLFilter(Filter):
  def apply(self, document):
    text = document.text
    text = re.sub(r"https?://\S+", "", text)
    document.text = text
    return document

class NewlineFilter(Filter):
  def apply(self, document):
    text = document.text
    # 1つ以上の連続した改行を表す正規表現
    regex = r"\n+"
    # 単一の改行
    replacement = "\n"
    # 引数3の文字列を対象に引数1を引数2で置き換える
    text = re.sub(regex, replacement, text)
    document.text = text
    return document

cleaner = Compose([
    URLFilter(),
    NewlineFilter()
])

MODELS = [
	"intfloat/multilingual-e5-base",
	"Qwen/Qwen3-Embedding-0.6B"
]

@st.cache_resource
def load_embedding_model():
	model_name = MODELS[0]
	embeddings = HuggingFaceEmbeddings(model_name=model_name)
	return embeddings

embeddings = load_embedding_model()

def tokenize(text):
	analyzer = Analyzer(token_filters=[POSKeepFilter(['名詞'])])
	nouns = [token.surface for token in analyzer.analyze(text)]
	return nouns

def extract(filepath):
	text = parser.from_file(filepath)
	assert isinstance(text, dict), f"tika.parser.from_file return {text}"
	return text

def get_file_list(path, extension):
	file_list = glob.glob(path + "/*." + extension)
	return file_list

def remove_url(text):
	assert isinstance(text, str)
	return re.sub(r"https?://\S+", "", text)

def remove_consecutive_newlines(text: str) -> str:
	# 1つ以上の連続した改行を表す正規表現
	regex = r"\n+"
	# 単一の改行
	replacement = "\n"
	# 引数3の文字列を対象に引数1を引数2で置き換える
	return re.sub(regex, replacement, text)

def clean_text(text):
	_text = remove_url(text)
	return remove_consecutive_newlines(_text)

def create_index_from_text(data_dir="./data"):
	docs = []
	for filepath in stqdm(get_file_list(data_dir, "txt")):
		with open(filepath) as f:
			doc = clean_text(f.read())
		docs.append(
			Document(
				page_content=doc,
				metadata=dict(
					url=filepath,
					kw=tokenize(doc)
				)
			)
		)
	return docs

def create_index_from_pdf(data_dir="./data"):
	docs = []
	for filepath in stqdm(get_file_list(data_dir, "pdf")):
		logger.info(f"indexing: {filepath}")
		doc = clean_text(extract(filepath).get("content"))
		docs.append(
			Document(
				page_content=doc,
				metadata=dict(
					url=filepath,
					kw=tokenize(doc)
				)
			)
		)
	logger.info("Finish convert rich text to normal text")
	return docs

def create_index_from_json(filepath="data/data.jsonl"):
	if not os.path.exists(filepath):
		st.warning("data.jsonl not found")
		return []
	with open(filepath) as f:
		data = [json.loads(line) for line in f.readlines()][:5000]
	docs = []
	for d in stqdm(data):
		doc = cleaner(d["Answer"])
		docs.append(
			Document(
				page_content=doc,
				metadata=dict(
					question=d["Question"],
					url=d["url"],
					copyright=d["copyright"],
					kw=tokenize(doc)
				)
			)
		)
	return docs

def create_index():
	logger.info("Creating index...")
	docs = []
	docs += create_index_from_text()
	docs += create_index_from_pdf()
	docs += create_index_from_json()
	logger.info("Creating index from files...")
	if len(docs) == 0:
		st.warning("No documents found for indexing.")
		return
	create_faiss_index(docs)

def create_faiss_index(docs, output_dir="./vector_store"):
	vector_store = FAISS.from_documents(
		docs,
		embeddings
	)
	vector_store.save_local(output_dir)
	st.metric("Indexed Docs", len(docs))
	st.success("Create index successfully")

def load_index(input_dir="./vector_store"):
	if not os.path.exists(input_dir):
		st.warning("index data not found")
		return None
	vector_store = FAISS.load_local(input_dir, embeddings, allow_dangerous_deserialization=True)
	return vector_store
