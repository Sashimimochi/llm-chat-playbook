# !/bin/bash

set -ue

MODEL_DIR=model
LOG_DIR=logs
DATA_DIR=data
MODEL=calm2-7b-chat.Q5_K_M.gguf
DICT=model_kwdlc.tar.gz
INDEX_DATA=data.jsonl

mkdir -p $LOG_DIR

if [ ! -d "$MODEL_DIR/kwdlc" ]; then
    wget https://github.com/lighttransport/jagger-python/releases/download/v0.1.0/$DICT
    tar -xvf $DICT
    rm $DICT
fi

if [ ! -f "$MODEL_DIR/$MODEL" ]; then
    wget https://huggingface.co/TheBloke/calm2-7B-chat-GGUF/resolve/main/$MODEL?download=true -O $MODEL
    mv $MODEL $MODEL_DIR/
fi

if [ ! -f "$DATA_DIR/$INDEX_DATA" ]; then
    wget https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k/resolve/main/data.jsonl?download=true -O $INDEX_DATA
    mv $INDEX_DATA $DATA_DIR/
fi
