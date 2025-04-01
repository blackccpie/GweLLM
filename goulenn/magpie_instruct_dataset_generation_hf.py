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

from litellm import text_completion, completion

#modelcard = 'huggingface/meta-llama/Meta-Llama-3-70B-Instruct'
#modelcard = 'huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1'
#pre_query_template_with_system_prompt = "<|start_header_id|>user<|end_header_id|>je suis un français, voici ma question: "

modelcard = "ollama/mistral-small:latest"
endpoint = 'http://10.55.8.5:11434'
pre_query_template_with_system_prompt = "[INST]je suis un français, voici ma question: "

#client = InferenceClient(model=modelcard, token=os.environ['HF_TOKEN_API'], headers={"X-use-cache": "false"})

questions = []
answers = []

# create QA pairs iteratively
for i in range(1):
    q_gen = text_completion(
        model=modelcard,
        api_base=endpoint,
        prompt=pre_query_template_with_system_prompt, 
        temperature=0.8, 
        stream=False,
        cache={"no-cache": True}
    ).choices[0].text
    question = extract_question(q_gen)
    print(f"clean:\n\033[96m{question}\033[0m")
    print("--")
    a_gen = completion(
        model=modelcard,
        api_base=endpoint,
        messages = [
            {
                "role": "user", 
                "content": 
                f"""répond directement à la question suivante en UNE SEULE phrase complète, en commençant directement par du texte.
        
                Par exemple: Quelle est la capitale de la France?
                Sortie attendue: La capitale de la France est Paris.

                Question: {question}
                """
            }
        ],
        temperature=0.5,
        stream=False,
        cache={"no-cache": True}
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