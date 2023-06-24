import transformers
print(transformers.__version__)
from configs import MODEL_ID, MAX_LENGTH, BNB_CONFIG, BATCH_SIZE
model_checkpoint = MODEL_ID

from datasets import load_dataset
from evaluate import load
import json

raw_datasets = load_dataset('json', data_files=['training_data.json', 'synth.json'])

raw_datasets = raw_datasets.map(lambda sample: {'output':json.dumps(sample['output'])})

"""The `dataset` object itself is [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key for the training, validation and test set:"""

raw_datasets

"""To access an actual element, you need to select a split first, then give an index:"""

print(raw_datasets["train"][0])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

prefix = "jsonconvert: "

"""We can then write the function that will preprocess our samples. We just feed them to the `tokenizer` with the argument `truncation=True`. This will ensure that an input longer that what the model selected can handle will be truncated to the maximum length accepted by the model. The padding will be dealt with later on (in a data collator) so we pad examples to the longest length in the batch and not the whole dataset."""

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)

    # Setup the tokenizer for targets
    outputs = [doc for doc in examples["output"]]
    labels = tokenizer(text_target=outputs, max_length=MAX_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print(preprocess_function(raw_datasets['train'][:2]))

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 128
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-xsum",
    # evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=3,
    num_train_epochs=100,
    predict_with_generate=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import nltk
import numpy as np

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model("t5-json-convert")

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load trained model and tokenizer
model_path = "t5-json-convert" 
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def generate_summary(text):
    # Add the prefix
    text = "jsonconvert: " + text

    # Encoding the inputs and passing them to model.generate()
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    outputs = model.generate(inputs, max_length=MAX_LENGTH, num_beams=5, early_stopping=True)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded_output

# Sample input for testing
sample_input = "can you please give me quote for 2.5mm usdt, I'll send CAD ?"

# Generate Summary
summary = generate_summary(sample_input)

print("result: ", summary)