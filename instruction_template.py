def template(input, output):
    INSTRUCTION = 'Q: Convert "{question}" into json format with keys side, baseAsset, quoteAsset, qty or value. JSON: {answer}'
    return INSTRUCTION.format(question=input, answer=output)