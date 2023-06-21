def template(input, output):
    INSTRUCTION = 'Can you extract the market information from the given prompt? Provide response in json format with keys baseAsset, quoteAsset, qty, value, side, ignore qty or value if it is not present {question} JSON: {answer}'
    return INSTRUCTION.format(question=input, answer=output)