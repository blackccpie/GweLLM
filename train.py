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

from datasets import Dataset, load_dataset

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

model_id = 'croissantllm/CroissantLLMBase'

############### PREPARE TOKENIZER

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id # Most LLMs don't have a pad token by default

############### PREPARE DATASET ###############

# saved in ~/.cache/huggingface/datasets
dataset = load_dataset("oscar-corpus/OSCAR-2201",
                        language="br", 
                        split="train")

print(dataset)

def tokenize_function(examples):
    # tokenize input text
    tokenized = tokenizer(examples["text"], padding="max_length", max_length=512, truncation=True)
    # set the labels to be the same as the input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=['input_ids', 'labels'])

tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

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
    model_id,
    device_map='auto',
    quantization_config=quant_config,
)
model.config.use_cache = False

# !!! should be called on the base model before loading the adapter, as it prepares the model’s layers to work with low-bit quantization !!!
model = prepare_model_for_kbit_training(model)

# load the PEFT model
model = get_peft_model(model,lora_config)

############### PEFT FINETUNING ###############

output_model_id = 'gwellm'

# set training arguments
training_args = TrainingArguments(
    report_to='none', # "codecarbon", "tensorboard", "wandb"
    overwrite_output_dir=True,
    output_dir=output_model_id,
    optim="paged_adamw_32bit",
    warmup_ratio=0.5,
    learning_rate=5e-5,
    num_train_epochs=1, # how many times will the same data be given
    per_device_train_batch_size=5, # how many data blocks to give the gpu per training step
    remove_unused_columns=False
)

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=collator
)

print("-> start training...")
trainer.train()

print("-> start post-evaluation...")
eval_results = trainer.evaluate()
print(f">>> perplexity: {math.exp(eval_results['eval_loss']):.2f}")



