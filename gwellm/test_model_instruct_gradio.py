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

# inference through HF transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# inference through llama.cpp
from llama_cpp import Llama

model_name = "gwellm-gemma2-2b-it-Q4_K_M-50k-e3.gguf"

def format_prompt_with_history(message, chat_history):
    # format the conversation history
    prompt = ""
    for interaction in chat_history:
        prompt += f"<start_of_turn>{interaction['role']}\n{interaction['content']}<end_of_turn>\n"

    # add the current user message
    prompt += f"<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>assistant\n"

    return prompt

class tf_inference:
    """
    GweLLM inference through HuggingFace Transformers
    """

    def __init__(self):
        """
        Constructor
        """
        # load the GweLLM model and tokenizer
        self.__tokenizer = AutoTokenizer.from_pretrained(".", gguf_file=model_name)
        self.__model = AutoModelForCausalLM.from_pretrained(".", gguf_file=model_name)

    def generate(self, message, chat_history):
        """
        Generates answer ton message
        """

        prompt = format_prompt_with_history(message, chat_history)

        # tokenize the input message
        inputs = self.__tokenizer(prompt, return_tensors="pt")

        # generate a response from the model
        outputs = self.__model.generate(**inputs)

        # decode the generated response
        return self.__tokenizer.decode(outputs[0], skip_special_tokens=True)

class lcpp_inference:
    """
    GweLLM inference through llama.cpp / llama-cpp-python bindings
    """

    def __init__(self):
        """
        Constructor
        """

        self.__llm = Llama(model_path=model_name, n_threads=4)  # Adjust n_threads as needed

    def generate(self, message, chat_history):
        """
        Generates answer ton message
        """

        prompt = format_prompt_with_history(message, chat_history)

        # generate a response from the model
        output = self.__llm(prompt, max_tokens=1024, stop=["\n"], stream=False) # TODO : better stop words

        # extract the generated response
        return output['choices'][0]['text'].strip()

#llm_engine = tf_inference()
llm_engine = lcpp_inference()

# maximum number of interactions to keep in history
max_history_length = 3

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="GweLLM Chatbot", type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        """
        Handles bot response generation
        """

        bot_message = llm_engine.generate(message, chat_history)
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})

        # limit the history length
        if len(chat_history) > max_history_length * 2:
            chat_history = chat_history[-max_history_length * 2:]

        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()