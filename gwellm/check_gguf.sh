#!/bin/bash

# retrieve model id
MODEL_ID=$(python3 -B model_library.py 'gemma2-2b')
echo "MODEL_ID: $MODEL_ID"

LLAMA_CPP_DIR=${LLAMA_CPP_DIR}
CUSTOM_VERSION='-110k-e4'
INPUT_GGUF_MODEL=$MODEL_ID-Q4_K_M$CUSTOM_VERSION.gguf

# run generation
$LLAMA_CPP_DIR/build/bin/llama-cli -m $INPUT_GGUF_MODEL -p "<start_of_turn>user\nPiv eo Albert Einstein?<end_of_turn>\n<start_of_turn>model\n" -n 256
