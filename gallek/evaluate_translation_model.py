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

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from datasets import load_dataset, concatenate_datasets, DatasetDict

import evaluate
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"setting device to: {device}")

source_lang='fr'
target_lang='br'

### step 1: load model and tokenizer

model_name = "gallek-m2m100-b40"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.src_lang = source_lang
tokenizer.tgt_lang = target_lang

### step 2: load dataset

# load dataset #1
ofis_dataset = load_dataset("Bretagne/ofis_publik_br-fr")
ofis_dataset = ofis_dataset.rename_column('français', 'fr')
ofis_dataset = ofis_dataset.rename_column('breton', 'br')

# load dataset #2
subtitles_dataset = load_dataset('Bretagne/OpenSubtitles_br_fr')

# load dataset #3
autogramm_dataset = load_dataset('Bretagne/Autogramm_Breton_translation')

# concatenate #1 & #2 & #3
dataset = DatasetDict({'train': concatenate_datasets([ofis_dataset['train'],subtitles_dataset['train'],autogramm_dataset['train']])})

print("loaded dataset infos:")
print(dataset)

dataset = dataset.shuffle(seeds=42)
dataset = dataset['train'].train_test_split(test_size=0.001)

### step 3: Generate translations
references = []  # list to hold reference translations
predictions = []  # list to hold model predictions

print(f"test size: {len(dataset['test'])}")

for sample in tqdm(dataset['test']):

    source_text = sample[source_lang]
    reference_text = sample[target_lang]
    
    # tokenize source text
    inputs = tokenizer(source_text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    # generate translation
    outputs = model.generate(**inputs, max_length=256) #, force_bos_token_id=tokenizer.get_lang_id(target_lang))
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # collect reference and prediction
    references.append(reference_text)  # sacrebleu expects a list of references for each sentence
    predictions.append(translated_text)

### step 4: Compute BLEU score

bleu = evaluate.load("sacrebleu")

# HF evaluate expects the references to be a list of lists (each list contains references for one example)
results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
print(results)
print(f"BLEU score: {results['score']}")