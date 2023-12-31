# -*- coding: utf-8 -*-
"""Sample sft trainer.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12ys6KfLvyKMQPDCcFDsk8q6v2v44JLBH
"""

# !pip install transformers bitsandbytes accelerate einops peft trl datasets

# !wget https://raw.githubusercontent.com/anandanand84/gpt2-train/main/training_data.json
# !wget https://raw.githubusercontent.com/anandanand84/gpt2-train/main/synth.json

from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, T5Tokenizer
from configs import MODEL_ID, MAX_LENGTH, BNB_CONFIG, BATCH_SIZE
from instruction_template import template, tokenizer, template_inference
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments
import json
from configs import MODEL_ID, MAX_LENGTH, BNB_CONFIG, BATCH_SIZE
from instruction_template import template



# Load dataset
def train():
    # ={ 'train' : 'training_data.json', 'test' : 'synth.json' 
    # tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    data = load_dataset('json', data_files=['training_data.json'])
    data = data.map(lambda sample: tokenizer(template(sample), truncation=True, padding='max_length', max_length=128))
    print(data['train'][0])
    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    # create TrainingArguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        logging_steps=50, 
        learning_rate=1e-4,
        save_steps=1000,
        per_device_train_batch_size=BATCH_SIZE,  # set batch size for training
        per_device_eval_batch_size=BATCH_SIZE,  # set batch size for evaluation
        # add other arguments as needed
    )

    trainer = Trainer(
        model=model,
        train_dataset=data['train'],
        args=training_args,  # the training arguments
    )

    trainer.train()
    trainer.save_model("sft-model")


def infer():
    model = AutoModelForCausalLM.from_pretrained('sft-model')
    # lora_config = LoraConfig.from_pretrained("sft-model")
    text = template_inference("<jsonconvert>give me a quote for 10 BTC?")
    tokens = tokenizer(text, return_tensors='pt')
    output = model.generate(**tokens, max_length=100, temperature=0.7, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)


train()
infer()