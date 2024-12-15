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

import context

from gallek.gallek_translator import gallek
from datasets import load_dataset

# saved in ~/.cache/huggingface/datasets
dataset = load_dataset( "jpacifico/French-Alpaca-dataset-Instruct-110K",
                        split="train")

small_dataset = dataset.take(10)

print(small_dataset)

# instanciate gallek translator
gk = gallek(chdir='../gallek/')

# define the translation functor
def to_br(sample):
  sample['instruction'] = gk.translate_fr2br(sample['instruction'])
  sample['input'] = gk.translate_fr2br(sample['input'])
  sample['output'] = gk.translate_fr2br(sample['output'])
  print(sample)

# translate the dataset to breton
small_dataset.map(to_br)

small_dataset.save_to_disk(dataset_path='alpaca-goulenn')
