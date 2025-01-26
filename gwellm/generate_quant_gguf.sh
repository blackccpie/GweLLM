#!/bin/bash

# retrieve model id
MODEL_ID=$(python3 -B model_library.py 'gemma2-2b')
echo "MODEL_ID: $MODEL_ID"

LLAMA_CPP_DIR=${LLAMA_CPP_DIR}
INPUT_HF_MODEL=$MODEL_ID-merged
OUTPUT_GGUF_MODEL=$MODEL_ID-f16.gguf
OUTPUT_TYPE=f16

# run conversion
python3 -B $LLAMA_CPP_DIR/convert_hf_to_gguf.py $INPUT_HF_MODEL --outfile $OUTPUT_GGUF_MODEL --outtype $OUTPUT_TYPE

INPUT_GGUF_MODEL=$OUTPUT_GGUF_MODEL
OUTPUT_GGUF_MODEL=$MODEL_ID-Q4_K_M.gguf
OUTPUT_TYPE=Q4_K_M

# run quantization
$LLAMA_CPP_DIR/build/bin/llama-quantize $INPUT_GGUF_MODEL $OUTPUT_GGUF_MODEL $OUTPUT_TYPE