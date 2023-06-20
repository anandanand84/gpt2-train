from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from flask import Flask, jsonify, request
import torch 
from instruction_template import template

model_id = "EleutherAI/gpt-j-6b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to(device)
model.eval()

app = Flask(__name__)

from peft import LoraConfig, get_peft_model
lora_config = LoraConfig.from_pretrained('outputs')
model = get_peft_model(model, lora_config)
print(model)

def infer(input):
    text = template(input)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.generate(**inputs,
        max_length=250, 
        top_k=0, 
        top_p=0.95, 
        num_return_sequences=1
    )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(result)
    return result

@app.route('/process', methods=['GET'])
def process_endpoint():
    query = request.args.get('query', default='', type=str)
    result = infer(query)
    return jsonify(result)

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the language model server!"

if __name__ == '__main__':
    input = "can you pls quote me 100 ADA, I'll send CAD"
    infer(input)
    app.run()