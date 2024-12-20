#!/bin/bash

LLAMA_CPP_DIR=${LLAMA_CPP_DIR}
INPUT_HF_MODEL=gwellm-it-merged
OUTPUT_GGUF_MODEL=gwellm-it-f16.gguf
OUTPUT_TYPE=f16

# run conversion
python3 -B $LLAMA_CPP_DIR/convert_hf_to_gguf.py $INPUT_HF_MODEL --outfile $OUTPUT_GGUF_MODEL --outtype $OUTPUT_TYPE

INPUT_GGUF_MODEL=$OUTPUT_GGUF_MODEL
OUTPUT_GGUF_MODEL=gwellm-it-Q4_K_M.gguf
OUTPUT_TYPE=Q4_K_M

# run conversion
$LLAMA_CPP_DIR/build/bin/llama-quantize $INPUT_GGUF_MODEL $OUTPUT_GGUF_MODEL $OUTPUT_TYPE