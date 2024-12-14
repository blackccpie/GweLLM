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

# inspired by: https://huggingface.co/docs/transformers/tasks/translation

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    pipeline
)

import evaluate

from datasets import load_dataset

import numpy as np


#source_lang='fr'
#target_lang='br'
#dataset = load_dataset("Bretagne/wikimedia_br_fr")

source_lang='français'
target_lang='breton'
dataset = load_dataset("Bretagne/ofis_publik_br-fr")

print("loaded dataset infos:")
print(dataset)

dataset = dataset['train'].train_test_split(test_size=0.2)

print(dataset["train"][0])

#checkpoint_base = "google-t5/t5-small"
checkpoint_base="facebook/m2m100_418M"
checkpoint = "my_awesome_breton_model"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_base)
tokenizer.src_lang = "fr"
tokenizer.tgt_lang = "br"

prefix = "traduis de français en breton: "

resume = False

# def preprocess_function(samples):
#     inputs = [prefix + sample[source_lang] for sample in samples]
#     targets = [sample[target_lang] for sample in samples]
#     model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
#     return model_inputs

def preprocess_function(sample):

    # TODO
    if not sample[source_lang] or not sample[target_lang]:
        print("invalid sample!")
        return {"input_ids": [], 'attention_mask': [], 'labels': []}
    
    #print(sample[source_lang])
    #print(sample[target_lang])

    input = prefix + sample[source_lang]
    target = sample[target_lang]
    model_input = tokenizer(input, text_target=target, max_length=128, truncation=True)
    return model_input

tokenized_dataset = dataset.map(preprocess_function, batched=False)

# filter out invalid samples
old_len = len(tokenized_dataset)
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)
new_len = len(tokenized_dataset)
print(f"removed {old_len-new_len} invalid samples from input dataset")

#print(tokenized_dataset["train"][0])
#print(tokenized_dataset["test"][0])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint_base, return_tensors="pt")

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint if resume else checkpoint_base)

training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint,
    overwrite_output_dir=True,
    report_to='none',
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True, #change to bf16=True for XPU
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(checkpoint)

#trainer.evaluate()

text = "traduis de français en breton: j'apprends le breton à l'école."

# Change `xx` to the language of the input and `yy` to the language of the desired output.
# Examples: "en" for English, "fr" for French, "de" for German, "es" for Spanish, "zh" for Chinese, etc; translation_en_to_fr translates English to French
# You can view all the lists of languages here - https://huggingface.co/languages
translator = pipeline("translation_fr_to_br", model="my_awesome_breton_model", device="cuda")
print(translator(text, max_length=256))