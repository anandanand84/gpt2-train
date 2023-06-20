"""First let's load the model we are going to use - GPT-neo-x-20B! Note that the model itself is around 40GB in half precision"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from instruction_template import template


#model_id = "EleutherAI/gpt-neox-20b"
model_id = "EleutherAI/gpt-j-6b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=True,
    bnb_4bit_use_double_quant=False
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

"""Let's load a common dataset, english quotes, to fine tune our model on famous quotes."""

from datasets import load_dataset


# data = load_dataset("Abirate/english_quotes")
# data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# data1 = load_dataset("Abirate/english_quotes")
# data1["train"][0]

INSTRUCTION = "Parse instruction below and extract information in json format with keys side, baseAsset, quoteAsset, qty or value."

from datasets import load_from_disk, load_dataset
import json
data = load_dataset('json', data_files=['training_data.json', 'synth.json'])
data = data.map(lambda sample: tokenizer(template(input=sample['input'], output=json.dumps(sample['output']))))

print(data["train"][0])


model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0})

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model
print(model)
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["k_proj", "v_proj", "q_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

"""Run the cell below to run the training! For the sake of the demo, we just ran it for few steps just to showcase how to use this integration with existing tools on the HF ecosystem."""

import transformers

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        max_steps=1000,
        learning_rate=0.01,
        logging_steps=4,
        output_dir="outputs",
        optim="adamw_torch"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("outputs")

lora_config = LoraConfig.from_pretrained('outputs')
model = get_peft_model(model, lora_config)

input = "can you pls quote me 100 ADA, I'll send CAD"
text = template(input, "")
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs,
        max_length=250, 
        top_k=0, 
        top_p=0.95, 
        num_return_sequences=1
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))