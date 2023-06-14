from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import json

INSTRUCTION = "<jsonconvert>"

BATCH_SIZE = 128

EPOCHS = 200

writer = SummaryWriter()

class TrainingData(Dataset):
    def __init__(self, path:str, synthpath:str, tokenizer):
        self.data = json.load(open(path, "r"))
        self.data_synth = json.load(open(synthpath, "r"))
        self.data.extend(self.data_synth)
        self.X = []
        print(len(self.data))
        for i, data in enumerate(self.data):
           self.X.append("<startofstring> " + INSTRUCTION + " " +data['input']+" <bot>: "+json.dumps(data['output'])+" <endofstring>")
        self.X = self.X[:20000]
        print(len(self.X))
        self.X_encoded = tokenizer(self.X, max_length=85, truncation=False, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])



from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
from flask import Flask, jsonify, request
app = Flask(__name__)


def train(training_data, model, optim):
    epochs = EPOCHS
    iterations = 0
    for i in tqdm.tqdm(range(epochs)):
        for X, a in training_data:
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
            iterations += 1
            writer.add_scalar('Loss/train', loss.item(), iterations)
        torch.save(model.state_dict(), "model_state.pt")
    print(infer("sell 100.25 worth of solana"))

def infer(inp):
    inp = "<startofstring> " +INSTRUCTION + " " +inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, max_length=60, 
        top_k=0, 
        top_p=0.95, 
        num_return_sequences=1)
    output = tokenizer.decode(output[0])
    return output


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)

# print(tokenizer.decode(model.generate(**tokenizer("hey i was good at basketball but ",
#                          return_tensors="pt"))[0]))

training_data = TrainingData("./training_data.json", "./synth.json", tokenizer)
# training_data = TrainingData("./training_data.json", "./training_data.json", tokenizer)
training_data =  DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

model.train()

optim = Adam(model.parameters(), lr=1e-3)

print("infer from model : ")
@app.route('/process', methods=['GET'])
def process_endpoint():
    query = request.args.get('query', default = '', type = str)
    result = infer(query)
    return jsonify(result)

print("training .... ")

train(training_data, model, optim)

app.run()

writer.close()
model.eval()

