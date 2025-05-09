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

import os

from datasets import Dataset

from magpie_instruct_processor import extract_question, extract_answer

from litellm import litellm, disable_cache

modelcard = 'huggingface/meta-llama/Llama-3.3-70B-Instruct'
#modelcard = 'huggingface/google/gemma-3-27b-it'
#modelcard = 'huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1'
endpoint = None

#modelcard = "ollama/mistral-small:latest"
#modelcard = "ollama/hf.co/croissantllm/CroissantLLMChat-v0.1-GGUF:Q8_0"
#endpoint = 'http://10.55.8.5:11434'

questions = []
answers = []

litellm.disable_cache()

# create QA pairs iteratively
for i in range(5):
    q_gen = litellm.completion(
        model=modelcard,
        api_base=endpoint,
        messages = [
            {
                "role": "user", 
                "content": 
                f"Propose moi un énoncé de question aléatoire pour la {i}ème question d'un quizz de culture générale, en répondant uniquement par la question."
            }
        ],
        temperature=0.8,
        stream=False,
        max_tokens=100,
        #extra_headers={"X-use-cache": "false"}, # TBC or HF Inference? (https://github.com/huggingface/huggingface_hub/issues/2081)
        cache={"no-cache": True}
    ).choices[0].message.content
    question = extract_question(q_gen)
    print(f"clean:\n\033[96m{question}\033[0m")
    print("--")
    a_gen = litellm.completion(
        model=modelcard,
        api_base=endpoint,
        messages = [
            {
                "role": "user", 
                "content": 
                f"""répond directement à la question suivante en UNE SEULE phrase complète, en commençant directement par du texte, sans paraphraser la question.

                Question: {question}
                """
            }
        ],
        temperature=0.5,
        stream=False,
    ).choices[0].message.content
    answer = extract_answer(a_gen)
    print(f"clean A:\n\033[92m{answer}\033[0m")
    print("------------------------")

    questions.append(question)
    answers.append(answer)

# build the dictionary
instruct_dictionary = {
    "input": questions,
    "output": answers
}

# build dataset and save
dataset = Dataset.from_dict(instruct_dictionary)
dataset.save_to_disk("my_dataset")