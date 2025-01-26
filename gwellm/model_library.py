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

class llama3_2_1b:
    base_model_id       = 'meta-llama/Llama-3.2-1B-Instruct'
    adapter_model_id    = 'gwellm-llama3.2-1b-it'
    start_pattern       = '<|start_header_id|>user<|end_header_id|>\n'
    next_pattern        = '<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n'
    end_pattern         = '<|eot_id|>'

class croissantllm:
    base_model_id       = 'croissantllm/CroissantLLMChat-v0.1'
    adapter_model_id    = 'croissantllmchat'
    start_pattern       = '<|im_start|>user\n'
    next_pattern        = '<|im_end|>\n<|im_start|>assistant\n'
    end_pattern         = '<|im_end|>'

class gemma2_2b:
    base_model_id       = 'google/gemma-2-2b-it'
    adapter_model_id    = 'gwellm-gemma2-2b-it'
    start_pattern       = '<start_of_turn>user\n'
    next_pattern        = '<end_of_turn>\n<start_of_turn>model\n'
    end_pattern         = '<end_of_turn>'

def get_model(name: str):
    """
    Retrieves model information class by name
    """
    if name == 'llama3.2-1b':
        return llama3_2_1b()
    elif name == 'croissantllm':
        return croissantllm()
    elif name == 'gemma2-2b':
        return gemma2_2b()