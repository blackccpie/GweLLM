# The MIT License

# Copyright (c) 2025 Albert Murienne

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import model_library

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

import torch

# load model card
modelcard = model_library.get_model('gemma2-2b')

# load tokenizer  base model
tokenizer = AutoTokenizer.from_pretrained(modelcard.base_model_id)
tokenizer.pad_token = tokenizer.eos_token # Most LLMs don't have a pad token by default
model = AutoModelForCausalLM.from_pretrained(modelcard.base_model_id, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16), device_map='auto')

# load adapter model
model.load_adapter(modelcard.adapter_model_id)
model.eval()
model.config.use_cache = True

with torch.no_grad():
    inputs = tokenizer([modelcard.test_query], add_special_tokens=True, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 256, do_sample=True, temperature=0.3, pad_token_id=tokenizer.pad_token_id)
    print(f"\n\nADAPTER MODEL OUTPUT:\n{tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)}")

