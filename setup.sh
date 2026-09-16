# !/bin/bash

set -ue

MODEL_DIR=model
LOG_DIR=logs
DATA_DIR=data
BASE_URL=https://huggingface.co
USERNAMES=(TheBloke mmnga)
REPO_NAMES=(calm2-7B-chat-GGUF cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-gguf)
MODELS=(calm2-7b-chat.Q5_K_M.gguf cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-Q4_K_M.gguf)
INDEX_DATA=data.jsonl

mkdir -p $LOG_DIR
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR

# Function download the model if it does not exist
function download_model() {
    BASE_URL=$1
    USERNAME=$2
    REPO_NAME=$3
    MODEL=$4
    if [ ! -f "$MODEL_DIR/$MODEL" ]; then
        wget $BASE_URL/$USERNAME/$REPO_NAME/resolve/main/$MODEL?download=true -O $MODEL
        mv $MODEL $MODEL_DIR/
    fi
}

# Download the all models
for i in "${!USERNAMES[@]}"; do
    download_model $BASE_URL ${USERNAMES[$i]} ${REPO_NAMES[$i]} ${MODELS[$i]}
done

if [ ! -f "$DATA_DIR/$INDEX_DATA" ]; then
    wget https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k/resolve/main/data.jsonl?download=true -O $INDEX_DATA
    mv $INDEX_DATA $DATA_DIR/
fi
