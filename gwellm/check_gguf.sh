#!/bin/bash

MODEL_CONFIG='gemma2-2b'

# retrieve model id
MODEL_ID=$(python3 -B model_library.py $MODEL_CONFIG name)
echo "MODEL_ID: $MODEL_ID"
# retrieve test query
TEST_QUERY=$(python3 -B model_library.py $MODEL_CONFIG test)
echo "MODEL_ID: $MODEL_ID"

LLAMA_CPP_DIR=${LLAMA_CPP_DIR}
CUSTOM_VERSION='-110k-e4'
INPUT_GGUF_MODEL=$MODEL_ID-Q4_K_M$CUSTOM_VERSION.gguf

# run generation
$LLAMA_CPP_DIR/build/bin/llama-cli -m $INPUT_GGUF_MODEL -p "$TEST_QUERY" -n 256
