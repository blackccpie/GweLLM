#!/bin/bash

LLAMA_CPP_DIR=${LLAMA_CPP_DIR}
INPUT_GGUF_MODEL=gwellm-gemma2-2b-it-Q4_K_M.gguf

# run generation
$LLAMA_CPP_DIR/build/bin/llama-cli -m $INPUT_GGUF_MODEL -p "<start_of_turn>user\nPiv eo Albert Einstein?<end_of_turn>\n<start_of_turn>model\n" -n 100
