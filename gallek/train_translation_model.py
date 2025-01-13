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

# inspired by: https://huggingface.co/docs/transformers/tasks/translation

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    pipeline
)

import evaluate

from datasets import load_dataset, concatenate_datasets, DatasetDict

import numpy as np

resume = False
revert = False # set tu True for backward br->fr translation training

source_lang='fr'
target_lang='br'
prefix = "traduis de français en breton: "
eval_query = prefix + "j'apprends le breton à l'école."
eval_pipeline = "translation_fr_to_br"

if revert:
    source_lang='br'
    target_lang='fr'
    prefix = "treiñ eus ar brezhoneg d'ar galleg: "
    eval_query = prefix + "deskiñ a ran brezhoneg er skol."
    eval_pipeline = "translation_br_to_fr"

# load dataset #1
ofis_dataset = load_dataset("Bretagne/ofis_publik_br-fr")
ofis_dataset = ofis_dataset.rename_column('français', 'fr')
ofis_dataset = ofis_dataset.rename_column('breton', 'br')

# load dataset #2
subtitles_dataset = load_dataset('Bretagne/OpenSubtitles_br_fr')

# concatenate #1 & #2
dataset = DatasetDict({'train': concatenate_datasets([ofis_dataset['train'],subtitles_dataset['train']])})

print("loaded dataset infos:")
print(dataset)

dataset = dataset.shuffle()
dataset = dataset['train'].train_test_split(test_size=0.2)

print(dataset["train"][0])

checkpoint_base="facebook/m2m100_418M"
checkpoint = "gallek-m2m100"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_base)
tokenizer.src_lang = source_lang
tokenizer.tgt_lang = target_lang

def preprocess_function(sample):
    """
    Tokenize sample
    """
    input = prefix + sample[source_lang]
    target = sample[target_lang]
    model_input = tokenizer(input, text_target=target, max_length=128, truncation=True)
    return model_input

def preprocess_function_batch(samples):
    """
    Tokenize samples (batch version)
    """
    inputs = [prefix + sample for sample in samples[source_lang]]
    targets = [sample for sample in samples[target_lang]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

#tokenized_dataset = dataset.map(preprocess_function, batched=False)
tokenized_dataset = dataset.map(preprocess_function_batch, batched=True, batch_size=16)

#print(tokenized_dataset["train"][0])
#print(tokenized_dataset["test"][0])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint_base, return_tensors="pt")

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    """
    Strips predictions
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    """
    Computes BLEU metric on given predictions
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # by default -100 is an index that is ignored in the loss function we use (Pytorch's cross entropy)
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

#generation_config = GenerationConfig(
#    decoder_start_token_id=model.config.decoder_start_token_id,
#    force_bos_token_id=tokenizer.get_lang_id(target_lang)
#)

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
    save_steps=500,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True, #change to bf16=True for XPU
    push_to_hub=False,
    #generation_config=generation_config
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

# start training
trainer.train()#resume_from_checkpoint = True)

# save model
trainer.save_model(checkpoint)

#trainer.evaluate()

# Change `xx` to the language of the input and `yy` to the language of the desired output.
# Examples: "en" for English, "fr" for French, "de" for German, "es" for Spanish, "zh" for Chinese, etc; translation_en_to_fr translates English to French
# You can view all the lists of languages here - https://huggingface.co/languages
translator = pipeline(eval_pipeline, model=checkpoint, device="cuda")
print(translator(eval_query, max_length=256))
