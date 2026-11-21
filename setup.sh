#!/usr/bin/env bash

set -eu

LOG_DIR=logs
DATA_DIR=data
INDEX_DATA=data.jsonl

mkdir -p "$LOG_DIR" "$DATA_DIR"

# Download the sample RAG dataset when it is not available locally.
if [ ! -f "$DATA_DIR/$INDEX_DATA" ]; then
    wget "https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k/resolve/main/data.jsonl?download=true" \
        -O "$DATA_DIR/$INDEX_DATA"
fi
