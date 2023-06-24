from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
from configs import MODEL_ID

START_OF_STRING = "<startofstring>"
END_OF_STRING = "<endofstring>"

tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
# tokenizer.add_special_tokens({"pad_token": "<pad>",  "bos_token": START_OF_STRING, "eos_token": END_OF_STRING})
# tokenizer.add_tokens(["<bot>:"])

###
# This is the template for the instruction should be called with a sample from the dataset containing input and output
def template(sample):
    INSTRUCTION = START_OF_STRING+" <jsonconvert> {question} <bot>: {answer} "+END_OF_STRING
    return INSTRUCTION.format(question=sample['input'], answer=sample['output'])

def template_inference(text):
    INSTRUCTION = START_OF_STRING+" <jsonconvert> {question} <bot>: "
    return INSTRUCTION.format(question=text)