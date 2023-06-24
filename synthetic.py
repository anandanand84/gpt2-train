import random
import json

def format_value(value):
    if value >= 1000000:
        return "{:.1f}mm".format(value / 1000000)
    elif value >= 10000:
        return "{:.1f}k".format(value / 1000)
    else:
        return str(value)

assets = ["CRO","CVX","SPELL","C98","ICP","SYS","LPT","HUSD","CBETH","STETH","FXS","MOB","ETC","AVAX","ALGO","ETH","USDT","GALA","OP","LUNC","WBTC","ALICE","AAVE","DOT","LDO","EOS","XLM","PAX","BUSD","ROSE","THETA","APE","MASK","XMR","XEC","ATOM","FTT","NEAR","CELO","QCAD","QNT","DAI","TUSD","CRV","SNX","KSM","YFI","RAY","BNT","ENS","CHZ","RNDR","CHR","DOGE","BAL","SHIB","BCH","LTC","ZEC","MIST","LINK","BNB","AION","EGLD","USD","ARDR","SAND","ONE","ANKR","AXS","LRC","FTM","FIL","1INCH","UMA","REN","BTC","ADA","TRX","LUNA2","ZRX","COMP","CAD","XRP","AMP","ZIL","ENJ","USDP","DYDX","BAND","CTSI","UNI","SUSHI","GRT","HOT","SRM","TULIP","PAXG","MANA","MKR","UST","XTZ","DASH","RUNE","MATIC","JPY","XEM","HNT","MIOTA","SKL","KNC","STORJ","SOL","HBAR","BSV","USDC","BAT","OMG","QTUM","LUNA","LEAG","ETHW","OSMO","TRAC","PLA","FLOKI","SNM","ARB","GBP","EUR","PEPE"]
quote_assets = ["USD", "CAD", "EUR", "GBP", "JPY"]

buy_qty_templates = [
    "Looking to long {qty} units of {asset}",
    "I am considering to buy {qty} units of {asset}",
    "I'd like a quote for {qty} units of {asset}, please",
    "Show me the price for {qty} units of {asset}",
    "Let's say I want {qty} units of {asset}, what's the price?",
    "What's the rate for {qty} units of {asset}?",
    "Could you show me how much {qty} units of {asset} would cost?",
    "How much would it cost to get {qty} units of {asset}?",
    "{asset} is what I want. How much for {qty} units?",
    "In {quote_asset}, what's the price of {qty} units of {asset}?",
    "I'm contemplating the purchase of {qty} units of {asset}",
    "What will be the cost for {qty} units of {asset}?",
    "How much would I need to invest for {qty} units of {asset}?",
    "What's the going rate for {qty} units of {asset}?",
    "What are {qty} units of {asset} going for these days?",
    "If I were to buy {qty} units of {asset}, what would be my total?",
    "What's the current value of {qty} units of {asset} in {quote_asset}?"
]

buy_value_templates = [
    "Looking to buy {asset} worth {value}",
    "I am considering to buy {asset} valued at {value}",
    "I'd like a quote for {asset} worth {value}, please",
    "Show me the price for {asset} worth {value}",
    "Let's say I want {asset} worth {value}, what's the price?",
    "What's the rate for {asset} valued at {value}?",
    "Could you show me how much {asset} worth {value} would cost?",
    "How much would it cost to get {asset} worth {value}?",
    "{asset} is what I want. How much for a total value of {value}?",
    "In {quote_asset}, what's the price of {asset} worth {value}?",
    "I'm contemplating a purchase of {asset} worth {value}",
    "What will {asset} costing {value} fetch me?",
    "I'm thinking of investing {value} in {asset}",
    "What can {value} get me in {asset}?",
    "I'm planning on buying {asset} with {value}",
    "How much {asset} can I get for {value} in {quote_asset}?"
]

sell_qty_templates = [
    "Looking to short {qty} units of {asset}",
    "I'm considering selling {qty} units of {asset}",
    "Can I have a quote for selling {qty} units of {asset}?",
    "What's the price for selling {qty} units of {asset}?",
    "I'm thinking of selling {qty} units of {asset}, how much will I get?",
    "How much {quote_asset} would I get for {qty} units of {asset}?",
    "What if I sold {qty} units of {asset}?",
    "Could you tell me how much I'd get for {qty} units of {asset}?",
    "I want to convert {qty} units of {asset} to {quote_asset}, how much would I get?",
    "{qty} units of {asset} for {quote_asset}, what's the exchange rate?",
    "I'm considering letting go of {qty} units of {asset}",
    "What would selling {qty} units of {asset} get me?",
    "If I decide to sell {qty} units of {asset}, what will be my return?",
    "How much {quote_asset} would selling {qty} units of {asset} bring in?",
    "What would be the proceeds from selling {qty} units of {asset}?",
    "What's the current value of {qty} units of {asset} if sold in {quote_asset}?",
    "Thinking about unloading {qty} units of {asset}, what's the likely return in {quote_asset}?"
]

sell_value_templates = [
    "I want to sell {asset} worth {value}",
    "I'm considering to sell {asset} valued at {value}",
    "Can I get a quote for selling {asset} worth {value}?",
    "What's the price for selling {asset} valued at {value}?",
    "I'm thinking of selling {asset} worth of {value}, how much should I send?",
    "What if I sold {asset} worth {value}?",
    "Could you tell me how much I'd get for {asset} worth {value}?",
    "{asset} worth {value} for {quote_asset}, what's the exchange rate?",
    "I have {asset} valued at {value} that I'd like to sell",
]



synthetic_data = []

def rounded_value(value):
    if value >= 1000000:
        value = round(value / 1000000) * 1000000  # Round to nearest million
        return value
    elif value >= 10000:
        value = round(value / 1000) * 1000  # Round to nearest thousand
        return value
    elif value >= 100:
        value = round(value / 100) * 100  # Round to nearest hundred
        return value
    else:
        return value

for i in range(1,5):
    for asset in assets:
        value = random.uniform(100, 6000000) 
        qty = value
        if random.random() < 0.12 and value > 10000:
            value = rounded_value(value)
            qty = value
            value_rounded = format_value(value)
            qty_rounded = format_value(qty)
        else:
            value_rounded = value
            qty_rounded = qty
        
        for template in buy_qty_templates:
            quote_asset = random.choice(quote_assets) if '{quote_asset}' in template else "USD"
            sentence = template.format(asset=asset, qty=qty_rounded, quote_asset=quote_asset)
            synthetic_data.append({"input": sentence, "output": {"baseAsset": asset, "side": "BUY", "qty": qty, "quoteAsset": quote_asset}})
        for template in buy_value_templates:
            quote_asset = random.choice(quote_assets) if '{quote_asset}' in template else "USD"
            sentence = template.format(asset=asset, value=value_rounded, quote_asset=quote_asset)
            synthetic_data.append({"input": sentence, "output": {"baseAsset": asset, "side": "BUY", "value": value, "quoteAsset": quote_asset}})
        for template in sell_qty_templates:
            quote_asset = random.choice(quote_assets) if '{quote_asset}' in template else "USD"
            sentence = template.format(asset=asset, qty=qty_rounded, quote_asset=quote_asset)
            synthetic_data.append({"input": sentence, "output": {"baseAsset": asset, "side": "SELL", "qty": qty, "quoteAsset": quote_asset}})
        for template in sell_value_templates:
            quote_asset = random.choice(quote_assets) if '{quote_asset}' in template else "USD"
            sentence = template.format(asset=asset, value=value_rounded, quote_asset=quote_asset)
            synthetic_data.append({"input": sentence, "output": {"baseAsset": asset, "side": "SELL", "value": value, "quoteAsset": quote_asset}})

            
synths = json.dumps(synthetic_data, indent=0)

with open("synth.json", "w") as f:
    f.write(synths + "\n")
