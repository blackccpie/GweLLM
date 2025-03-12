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
import re

from datasets import Dataset

modelcard = 'meta-llama/Meta-Llama-3-70B-Instruct'
pre_query_template_with_system_prompt = "<|start_header_id|>user<|end_header_id|>je suis un français, voici ma question: "

from huggingface_hub import InferenceClient

client = InferenceClient(model=modelcard, token=os.environ['HF_TOKEN_API'], headers={"X-use-cache": "false"})

def clean_question(question):
    """
    Clean the question from unwanted formatting
    """
    # regex explanation:
    # ^\d+[\).]?\s* → removes leading numbers (e.g., "1)", "5.", etc.) and spaces
    # ^["«]* → removes leading quotes ("), guillemets («), and dash (-)
    question = re.sub(r'^(?:\d+[\).]?\s*)?["«-]*', '', question).strip()
    # ensure the first letter is capitalized
    return question[0].upper() + question[1:]

def extract_question(model_output):
    """
    Extract the question from model generation
    """
    try:
        extract_question = f"{model_output.split('?')[0].strip()}?"
        print(rf"extracted Q: {extract_question}")
        return clean_question(extract_question)
    except:
        print("failed extracting question")
        return None

def clean_answer(model_output):
    """
    Clean the answer from unwanted formatting
    """
    # use regex to remove leading spaces and a colon (if present), along with spaces after the colon.
    answer = re.sub(r'^\s*:\s*', '', model_output).strip()
    return answer

def extract_answer(model_output):
    """
    Clean the question from leading unwanted formatting
    """
    try:
        extract_answer = f"{model_output.split('.')[0].strip()}."
        print(rf"extracted A: {extract_answer}")
        return clean_answer(extract_answer)
    except:
        print("failed extracting answer")
        return None

questions = []
answers = []

# create QA pairs iteratively
for i in range(10):
    q_gen = client.text_generation(pre_query_template_with_system_prompt, do_sample=True, temperature=0.8)
    question = extract_question(q_gen)
    print(f"clean:\n\033[96m{question}\033[0m")
    print("--")
    a_gen = client.chat_completion(
        [
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
        temperature=0.5
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