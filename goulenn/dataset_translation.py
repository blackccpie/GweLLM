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
from datasets import load_dataset, DatasetInfo

new_dataset_name_prefix = 'goulenn-alpaca'

# saved in ~/.cache/huggingface/datasets
dataset = load_dataset( path="jpacifico/French-Alpaca-dataset-Instruct-110K",
                        split="train")

# subsample dataset
subset_size = 1
small_dataset = dataset.take(subset_size)
new_dataset_name = f"{new_dataset_name_prefix}-{subset_size}"

print(small_dataset)

# instanciate gallek translator
gk = gallek(chdir='../gallek/', max_length=600)

# define the translation functor
def to_br(sample):
  sample['instruction'] = gk.translate_fr2br(sample['instruction'])
  sample['input'] = gk.translate_fr2br(sample['input'])
  sample['output'] = gk.translate_fr2br(sample['output'])
  #print(sample)
  return sample

# translate the dataset to breton
small_dataset = small_dataset.map(to_br, batched=False)

#print(small_dataset[0])

# access the DatasetInfo object
info = DatasetInfo()

# Modify DatasetInfo fields
info.description = f"{subset_size} Breton instructions dataset, translated from the original French-Alpaca-dataset-Instruct-110K by Jonathan Pacifico."
info.homepage = "https://github.com/blackccpie/GweLLM"
info.license = "MIT License"
info.version = "1.0.0"

small_dataset.save_to_disk(dataset_path=new_dataset_name)

info.write_to_directory(new_dataset_name, pretty_print=True)
