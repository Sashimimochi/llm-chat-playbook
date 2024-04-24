# !/bin/bash

set -ue

MODEL_DIR=model
LOG_DIR=logs
MODEL=calm2-7b-chat.Q5_K_M.gguf
DICT=model_kwdlc.tar.gz

mkdir -p $LOG_DIR

if [ ! -d "$MODEL_DIR/kwdlc" ]; then
    wget https://github.com/lighttransport/jagger-python/releases/download/v0.1.0/$DICT
    tar -xvf $DICT
fi
if [ ! -f "$MODEL_DIR/$MODEL" ]; then
    wget https://huggingface.co/TheBloke/calm2-7B-chat-GGUF/resolve/main/$MODEL?download=true -O $MODEL
    mv $MODEL $MODEL_DIR/
fi
