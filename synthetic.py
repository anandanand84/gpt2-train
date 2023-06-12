import random
import json

assets = ["CRO","CVX","SPELL","C98","ICP","SYS","LPT","HUSD","CBETH","STETH","FXS","MOB","ETC","AVAX","ALGO","ETH","USDT","GALA","OP","LUNC","WBTC","ALICE","AAVE","DOT","LDO","EOS","XLM","PAX","BUSD","ROSE","THETA","APE","MASK","XMR","XEC","ATOM","FTT","NEAR","CELO","QCAD","QNT","DAI","TUSD","CRV","SNX","KSM","YFI","RAY","BNT","ENS","CHZ","RNDR","CHR","DOGE","BAL","SHIB","BCH","LTC","ZEC","MIST","LINK","BNB","AION","EGLD","USD","ARDR","SAND","ONE","ANKR","AXS","LRC","FTM","FIL","1INCH","UMA","REN","BTC","ADA","TRX","LUNA2","ZRX","COMP","CAD","XRP","AMP","ZIL","ENJ","USDP","DYDX","BAND","CTSI","UNI","SUSHI","GRT","HOT","SRM","TULIP","PAXG","MANA","MKR","UST","XTZ","DASH","RUNE","MATIC","JPY","XEM","HNT","MIOTA","SKL","KNC","STORJ","SOL","HBAR","BSV","USDC","BAT","OMG","QTUM","LUNA","LEAG","ETHW","OSMO","TRAC","PLA","FLOKI","SNM","ARB","GBP","EUR","PEPE"]
quote_assets = ["USD", "CAD", "EUR", "GBP", "JPY"]

buy_qty_templates = [
    "I'd like to purchase {} units of {}",
    "Can I get a quote for buying {} units of {}",
    "What if I wanted to secure {} units of {}",
    "Assuming I want to acquire {} units of {}",
    "I'm interested in obtaining {} units of {}",
    "Can you provide me with a quote for {} units of {}",
    "I wish to buy {} units of {}",
    "I'm looking to acquire {} units of {}, priced in {}",
    "Could I buy {} units of {}, using {} as the quote asset?"
]

buy_value_templates = [
    "I'd like to purchase {} worth of {}",
    "Can I get a quote for buying {} worth of {}",
    "What if I wanted to secure {} worth of {}",
    "Assuming I want to acquire {} worth of {}",
    "I'm interested in obtaining {} worth of {}",
    "Can you provide me with a quote for {} worth of {}",
    "I wish to buy {} worth of {}",
    "I'm looking to secure {} worth of {}, priced in {}",
    "Could I buy {} worth of {}, with {} as the quote asset?"
]

sell_qty_templates = [
    "What would be the outcome if I sold {} units of {}",
    "If I sold {} units of {}, how much could I expect in return",
    "Looking to get rid of {} units of {}. What's the current rate?",
    "Can I get a quote for selling {} units of {}",
    "I'm thinking about selling {} units of {}. How much would that be?",
    "I'd like to sell {} units of {}. What would be the price?",
    "Planning on selling {} units of {}. What would the quote be?",
    "I'd like to sell {} units of {}, converting to {}",
    "Could I sell {} units of {}, and receive {} in return?"
]

sell_value_templates = [
    "What would be the outcome if I sold {} worth of {}",
    "If I sold {} worth of {}, how much could I expect in return",
    "Looking to get rid of {} worth of {}. What's the current rate?",
    "Can I get a quote for selling {} worth of {}",
    "I'm thinking about selling {} worth of {}. How much would that be?",
    "I'd like to sell {} worth of {}. What would be the price?",
    "Planning on selling some of my {}. Specifically {} worth. What would the quote be?",
    "I'd like to sell {} worth of {}, converting to {}",
    "Could I sell {} worth of {}, and receive {} in return?"
]


synthetic_data = []

for i in range(20):
    for asset in assets:
        value = random.uniform(100, 100000) 
        qty = random.uniform(1, 100000)
        for template in buy_qty_templates:
            quote_asset = "USD" if len(template.split("{}")) == 3 else random.choice(quote_assets)
            sentence = template.format(qty, asset, quote_asset)
            synthetic_data.append({"input": sentence, "output": {"baseAsset": asset, "side": "BUY", "qty": qty, "quoteAsset": quote_asset}})
        for template in buy_value_templates:
            quote_asset = "USD" if len(template.split("{}")) == 3 else random.choice(quote_assets)
            sentence = template.format(value, asset, quote_asset)
            synthetic_data.append({"input": sentence, "output": {"baseAsset": asset, "side": "BUY", "value": value, "quoteAsset": quote_asset}})
        for template in sell_qty_templates:
            quote_asset = "USD" if len(template.split("{}")) == 3 else random.choice(quote_assets)
            sentence = template.format(qty, asset, quote_asset)
            synthetic_data.append({"input": sentence, "output": {"baseAsset": asset, "side": "SELL", "qty": qty, "quoteAsset": quote_asset}})
        for template in sell_value_templates:
            quote_asset = "USD" if len(template.split("{}")) == 3 else random.choice(quote_assets)
            sentence = template.format(value, asset, quote_asset)
            synthetic_data.append({"input": sentence, "output": {"baseAsset": asset, "side": "SELL", "value": value, "quoteAsset": quote_asset}})

            
synths = json.dumps(synthetic_data, indent=0)

with open("synth.json", "w") as f:
    f.write(synths + "\n")
