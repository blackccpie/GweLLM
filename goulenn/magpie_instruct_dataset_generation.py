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

# This is a custom implementation othe "Magpie" alignment data synthesis technique:
# https://magpie-align.github.io/

# inference through HF transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import re

modelcard = 'croissantllm/CroissantLLMChat-v0.1'
pre_query_template_with_system_prompt = "<|im_start|>user\nje suis un français, voici ma question: "

# load the GweLLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelcard, use_fast=True, legacy=False)
model = AutoModelForCausalLM.from_pretrained(modelcard, device_map='auto')

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, do_sample=True, temperature=0.9, truncation=True, max_length=100, return_full_text=False)

def clean_question(question):
    """
    TODO
    """
    # remove number prefixes and optional punctuation after them
    question = re.sub(r"^\d+[\).]?\s*", "", question)
     # remove leading quotation marks if present
    question = question.lstrip('"')
    # ensure the first letter is capitalized
    return question[0].upper() + question[1:]

def extract_question(answer):
    """
    TODO
    """
    try:
        return clean_question(f"{answer.split('?')[0].strip()}?")
    except:
        print("failed extraction question from answer")

for i in range(10):
    answer = pipe(pre_query_template_with_system_prompt, chat_template=None)[0]['generated_text']
    print(extract_question(answer))
    print("------------------------")

