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

fw_modelcard = "gallek-m2m100-b40"
bw_modelcard = "kellag-m2m100-b51"

fw_model = AutoModelForSeq2SeqLM.from_pretrained(fw_modelcard)
fw_tokenizer = AutoTokenizer.from_pretrained(fw_modelcard)

fw_translation_pipeline = pipeline("translation", model=fw_model, tokenizer=fw_tokenizer, src_lang='fr', tgt_lang='br', max_length=400, device="cuda")

bw_model = AutoModelForSeq2SeqLM.from_pretrained(bw_modelcard)
bw_tokenizer = AutoTokenizer.from_pretrained(bw_modelcard)

bw_translation_pipeline = pipeline("translation", model=bw_model, tokenizer=bw_tokenizer, src_lang='br', tgt_lang='fr', max_length=400, device="cuda")

# translation function
def translate(text, direction):
    if direction == "fr_to_br":
        return fw_translation_pipeline("traduis de français en breton: " + text)[0]['translation_text']
    else:
        return bw_translation_pipeline("treiñ eus ar galleg d'ar brezhoneg: " + text)[0]['translation_text']

# function to switch translation direction
def switch_direction(direction):
    return "br_to_fr" if direction == "fr_to_br" else "fr_to_br"

# function to update labels dynamically
def update_labels(direction):
    if direction == "br_to_fr":
        return gr.Textbox(label="Breton"), gr.Textbox(label="French")
    else:
        return gr.Textbox(label="French"), gr.Textbox(label="Breton")

with gr.Blocks() as demo:

    gr.Markdown("# Gallek French <-> Breton Translation Demo")

    direction = gr.State("fr_to_br")  # default direction is French to Breton
    
    input_text = gr.Textbox(label="French")
    output_text = gr.Textbox(label="Breton")
    
    with gr.Row():
        translate_btn = gr.Button("Translate", variant='primary')
        switch_btn = gr.Button("Switch Direction", variant='secondary')

    # translation logic
    translate_btn.click(translate, [input_text, direction], output_text)

    # switch direction logic
    switch_btn.click(switch_direction, direction, direction).then(
        update_labels, direction, [input_text, output_text]
    )

demo.launch()