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

from datasets import Dataset

from magpie_instruct_processor import extract_question, extract_answer

from tqdm import tqdm

modelcard = 'croissantllm/CroissantLLMChat-v0.1'
pre_query_template_with_system_prompt = "<|im_start|>user\nje suis un français, voici ma question: "

# load the GweLLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelcard, use_fast=True, legacy=False)
model = AutoModelForCausalLM.from_pretrained(modelcard, device_map='auto')
model.eval()
model.config.use_cache = True

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, do_sample=True, temperature=0.5, truncation=True, max_new_tokens=100, return_full_text=False)

questions = []
answers = []

# create QA pairs iteratively
for i in tqdm(range(1000), unit="QA", desc="Generating QA pairs"):
    q_gen = pipe(pre_query_template_with_system_prompt, chat_template=None)[0]['generated_text']
    question = extract_question(q_gen)
    print(f"clean:\n\033[96m{question}\033[0m")
    print("--")
    a_gen = pipe(
        f"""<|im_start|>user
        répond directement à la question suivante en UNE SEULE phrase complète, en commençant directement par du texte, sans paraphraser la question.
        
        Par exemple: Quelle est la capitale de la France?
        Sortie attendue: Il s'agit de Paris.

        Question: {question}
        <|im_end|><|im_start|>assistant"""
        )[0]['generated_text']
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