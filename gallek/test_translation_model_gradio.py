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

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

#modelcard = "facebook/m2m100_418M"
modelcard = "gallek-m2m100-b33"

model = AutoModelForSeq2SeqLM.from_pretrained(modelcard)
tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

def translate(text):
    """
    Translate the text from source lang to target lang
    """
    translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, src_lang='fr', tgt_lang='br', max_length=400, device="cuda")
    result = translation_pipeline(text)
    return result[0]['translation_text']

demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.components.Textbox(label="Text"),
    ],
    outputs=["text"],
    cache_examples=False,
    title="Translation Demo",
    allow_flagging='never'
)

demo.launch()