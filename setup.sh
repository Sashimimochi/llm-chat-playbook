# !/bin/bash

set -ue

mkdir -p logs
wget https://github.com/lighttransport/jagger-python/releases/download/v0.1.0/model_kwdlc.tar.gz
tar -xvf model_kwdlc.tar.gz
wget https://huggingface.co/TheBloke/calm2-7B-chat-GGUF/resolve/main/calm2-7b-chat.Q5_K_M.gguf?download=true
mv calm2-7b-chat.Q5_K_M.gguf model/
