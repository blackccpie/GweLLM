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

batch_size = 16

# saved in ~/.cache/huggingface/datasets
dataset = load_dataset( path="jpacifico/French-Alpaca-dataset-Instruct-110K",
                        split="train")

# subsample dataset
new_dataset_name = ""

# -- skip items --
skip_size = 0
small_dataset = dataset.skip(skip_size)

# -- subset --
subset_size = 50000 # if 0, takes all samples
if subset_size == 0:
    subset_size = small_dataset.num_rows
else:
    small_dataset = dataset.take(subset_size)

new_dataset_name = f"{new_dataset_name_prefix}-{subset_size}-skip{skip_size}"

print(f"dataset name: {new_dataset_name}")
print(small_dataset)

# instanciate gallek translator
gk = gallek(chdir='../gallek/', max_length=600, batch_size=batch_size)

def to_br_batch(samples):
    """
    Translates all fields in batches into Breton
    """

    instructions = gk.translate_fr2br_batch(samples['instruction'])
    inputs = gk.translate_fr2br_batch(samples['input'])
    outputs = gk.translate_fr2br_batch(samples['output'])

    samples['instruction'] = [t['translation_text'] for t in instructions]
    samples['input'] = [t['translation_text'] for t in inputs]
    samples['output'] = [t['translation_text'] for t in outputs]
 
    return samples

# translate the dataset to breton
small_dataset = small_dataset.map(to_br_batch, batched=True, batch_size=batch_size, writer_batch_size=100)

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
