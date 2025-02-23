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

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)

import torch

# CHAT MODEL

chat_model_id = "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF"
chat_gguf = "Llama-3.2-3B-Instruct.Q4_K_M.gguf"

tokenizer = AutoTokenizer.from_pretrained(chat_model_id, gguf_file=chat_gguf)
model = AutoModelForCausalLM.from_pretrained(chat_model_id, gguf_file=chat_gguf)

chat_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, do_sample=True, temperature=0.5, truncation=True, max_length=512, return_full_text=False)

# TRANSLATION MODELS

fw_modelcard = "../gallek/gallek-m2m100-b51"
bw_modelcard = "../gallek/kellag-m2m100-b51"

fw_model = AutoModelForSeq2SeqLM.from_pretrained(fw_modelcard)
fw_tokenizer = AutoTokenizer.from_pretrained(fw_modelcard)

fw_translation_pipeline = pipeline("translation", model=fw_model, tokenizer=fw_tokenizer, src_lang='fr', tgt_lang='br', max_length=400, device="cuda")

bw_model = AutoModelForSeq2SeqLM.from_pretrained(bw_modelcard)
bw_tokenizer = AutoTokenizer.from_pretrained(bw_modelcard)

bw_translation_pipeline = pipeline("translation", model=bw_model, tokenizer=bw_tokenizer, src_lang='br', tgt_lang='fr', max_length=400, device="cuda")

# translation function
def translate(text, forward: bool):
    if forward:
        return fw_translation_pipeline("traduis de français en breton: " + text)[0]['translation_text']
    else:
        return bw_translation_pipeline("treiñ eus ar galleg d'ar brezhoneg: " + text)[0]['translation_text']

# answer function
def answer(text):
     return chat_pipeline(text, chat_template=None)[0]['generated_text']

def format_prompt_with_history(message, native_chat_history):
    # format the conversation history
    prompt = ""
    for interaction in native_chat_history:
        prompt += f"<|start_header_id|>{interaction['role']}<|end_header_id|>\n{interaction['content']}<|eot_id|>\n"

    # add the current user message
    prompt += f"<|start_header_id|>user<|end_header_id|>\ntu es un assistant francophone. Répond en une seule phrase sans formattage.\n{message}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"

    return prompt

# maximum number of interactions to keep in history
max_history_length = 3

# keep a hidden model "native" language chat history
native_chat_history = []

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    chatbot = gr.Chatbot(label="Breton Chatbot (Translation based)", type="messages")
    msg = gr.Textbox(label='User Input')

    def clear(chat_history):
        """
        Handles clearing chat
        """
        chat_history.clear()
        native_chat_history.clear()

    chatbot.clear(clear, inputs=[chatbot])

    def respond(message, chat_history):
        """
        Handles bot response generation
        """

        global native_chat_history

        fr_message = translate(message, forward=False)
        print(f"user fr -> {fr_message}")

        prompt = format_prompt_with_history(fr_message, native_chat_history)

        bot_fr_message = answer(prompt)
        print(f"bot fr -> {bot_fr_message}")
        bot_br_message = translate( bot_fr_message, forward=True)
        print(f"bot br -> {bot_br_message}")
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_br_message})

        native_chat_history.append({"role": "user", "content": fr_message})
        native_chat_history.append({"role": "assistant", "content": bot_fr_message})

        # limit the history length
        if len(chat_history) > max_history_length * 2:
            chat_history = chat_history[-max_history_length * 2:]
            native_chat_history = native_chat_history[-max_history_length * 2:]

        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()