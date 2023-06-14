import json

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tqdm
import torch
from flask import Flask, jsonify, request

INSTRUCTION = "<jsonconvert>"

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load("model_state.pt", map_location=device))
model = model.to(device)

model.eval()

def infer(inp):
    inp = "<startofstring> " +INSTRUCTION + " " +inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a,
        max_length=60, 
        top_k=0, 
        top_p=0.95, 
        num_return_sequences=1
    )
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output)
    # take stuff after <bot>:
    output = output.split("<bot>: ")
    if len(output) > 1:
        output = output[1]
        output = json.loads(output)
    else:
        output = {}
    return output


@app.route('/process', methods=['GET'])
def process_endpoint():
    query = request.args.get('query', default='', type=str)
    result = infer(query)
    return jsonify(result)

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the language model server!"

if __name__ == '__main__':
    app.run()