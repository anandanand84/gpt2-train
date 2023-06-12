import json
data = json.load(open('training_data.json', "r"))

with open("training_data.jsonl", "w") as f:
    for i, item in enumerate(data):
        print(item)
        line = { "input_text": item['input'], "output_text": json.dumps(item['output']) }
        f.write(json.dumps(line) + "\n")
        print(line)