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

import model_library

from datasets import load_from_disk, load_dataset

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

import math
import torch

# load model card
modelcard = model_library.get_model('gemma2-2b')

force_pretraining = False
resume = False

############### PREPARE TOKENIZER

tokenizer = AutoTokenizer.from_pretrained(modelcard.base_model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id # Most LLMs don't have a pad token by default # TODO confirm & clarify?

############### PREPARE DATASET ###############

def tokenize_instruct_dataset():
    """
    Loads and tokenizes instruct dataset
    """

    # saved in ~/.cache/huggingface/datasets
    dataset = load_from_disk("../goulenn/goulenn-alpaca-110000")

    print("loaded dataset infos:")
    print(dataset)

    def generate_prompt(sample):

        # samples with additional context into
        if sample['input']:
            text = f"""{modelcard.start_pattern}{sample["instruction"]}\n\n{sample["input"]}{modelcard.next_pattern}{sample["output"]}{modelcard.end_pattern}"""
        # without
        else:
            text = f"""{modelcard.start_pattern}{sample["instruction"]}{modelcard.next_pattern}{sample["output"]}{modelcard.end_pattern}"""
        return text

    # add the "prompt" column in the dataset
    text_column = [generate_prompt(data_point) for data_point in dataset]
    dataset = dataset.add_column("prompt", text_column)
    dataset = dataset.shuffle() #seed=1234

    def tokenize_function(samples):
        # tokenize input text
        tokenized = tokenizer(samples["prompt"], padding="max_length", max_length=512, truncation=True)
        # set the labels to be the same as the input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=['input_ids', 'labels']) # TODO : why is that mandatory???? correct further down, and also in sandboxed train_model.py
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    #print(tokenized_dataset['train'][0])

    return tokenized_dataset

def tokenize_pretraining_dataset():
    """
    Loads and tokenizes causal pretraining dataset
    """

    # load dataset
    dataset = load_dataset("Bretagne/WikiMatrix_br")
    dataset = dataset.shuffle() #seed=1234

    print("loaded dataset infos:")
    print(dataset)

    def tokenize_function(samples):
        # tokenize input text
        tokenized = tokenizer(samples["text"], padding="max_length", max_length=128, truncation=True)
        # set the labels to be the same as the input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=['input_ids', 'labels'])
    tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.2)

    #print(tokenized_dataset['train'][0])

    return tokenized_dataset

tokenized_dataset = None

if force_pretraining:
    tokenized_dataset = tokenize_pretraining_dataset()
else:
    tokenized_dataset = tokenize_instruct_dataset()

############### PREPARE MODEL ###############

# data collator for dynamic padding
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)

# load quantized model
model = AutoModelForCausalLM.from_pretrained(
    modelcard.base_model_id,
    device_map='auto',
    quantization_config=quant_config,
)
model.config.use_cache = False

# !!! should be called on the base model before loading the adapter, as it prepares the model’s layers to work with low-bit quantization !!!
model = prepare_model_for_kbit_training(model)

# load the PEFT model
if not resume:
    model = get_peft_model(model,lora_config)
else:
    model = PeftModel.from_pretrained(
                model, 
                modelcard.adapter_model_id, 
                is_trainable=True, 
                quantization_config=quant_config,
                device_map='auto')

############### PEFT FINETUNING ###############

# set training arguments
training_args = TrainingArguments(
    report_to='none', # "codecarbon", "tensorboard", "wandb"
    overwrite_output_dir=True,
    output_dir=modelcard.adapter_model_id,
    optim="paged_adamw_32bit",
    warmup_ratio=0.5,
    learning_rate=5e-5,
    num_train_epochs=1, # how many times will the same data be given
    per_device_train_batch_size=16, # how many data blocks to give the gpu per training step
    per_device_eval_batch_size=16, # how many data blocks to give the gpu per testing step
    remove_unused_columns=False
)

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=collator
)

torch.cuda.empty_cache()

print("-> start training...")
trainer.train() #resume_from_checkpoint = True

print("-> start post-evaluation...")
eval_results = trainer.evaluate()
print(f">>> perplexity: {math.exp(eval_results['eval_loss']):.2f}")

print("-> saving model...")
trainer.save_model(modelcard.adapter_model_id)


