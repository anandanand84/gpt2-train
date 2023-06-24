from transformers import BitsAndBytesConfig

MODEL_ID = "t5-base"

MAX_LENGTH = 128

BATCH_SIZE = 32

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True
)
