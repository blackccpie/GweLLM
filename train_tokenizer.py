# The MIT License

# Copyright (c) 2017-2017 Albert Murienne

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

# inspired by : https://huggingface.co/learn/nlp-course/chapter6/2

from datasets import load_dataset

from transformers import (
    AutoTokenizer
)

model_id = 'croissantllm/CroissantLLMBase'

example = "N'eus ket bet trawalc'h ?"

############### PREPARE DATASET ###############

# saved in ~/.cache/huggingface/datasets
dataset = load_dataset("oscar-corpus/OSCAR-2201",
                        language="br", 
                        split="train")

print(dataset)
print(dataset[100]['text'])

def get_training_corpus():
    """
    Returns a python generator (avoid loading into memory until necessary) to load 1000 texts at a time
    """
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["text"]

training_corpus = get_training_corpus()

############### PREPARE TOKENIZER

old_tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"old tokenizer vocabulary size is: {len(old_tokenizer)}")

print(old_tokenizer.tokenize(example))

############### TRAIN TOKENIZER

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=32000)

print(tokenizer.tokenize(example))

############### SAVE TOKENIZER

tokenizer.save_pretrained("gwellm-tokenizer")
