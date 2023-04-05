from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(
    "swype/deepshard-13b-raw",
    use_auth_token="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"
)