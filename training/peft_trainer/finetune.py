import os
import sys

import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from dataset import HF_TOKEN, make_supervised_data_module, smart_tokenizer_and_embedding_resize, special_tokens


# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  # we don't always need 3 tbh
LEARNING_RATE = 2e-5  # the Karpathy constant
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
OUTPUT_DIR = "lora-alpaca"
HF_TOKEN = os.environ.get("HF_TOKEN")

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size


model = LlamaForCausalLM.from_pretrained(
    "swype/deepshard-13B-raw",
    use_auth_token=HF_TOKEN,
    load_in_8bit=True,
    device_map=device_map,
)
tokenizer = LlamaTokenizer.from_pretrained(
    "swype/deepshard-13B-raw",
    use_auth_token=HF_TOKEN,
)

smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens,
    tokenizer=tokenizer,
    model=model,
)
data_module = make_supervised_data_module(tokenizer=tokenizer)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

trainer = transformers.Trainer(
    model=model,
    **data_module,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=0.03,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=5,
        evaluation_strategy="no",
        save_strategy="steps",
        eval_steps=None,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if ddp else None,
        lr_scheduler_type="cosine",
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap='LlamaDecoderLayer'
    )
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train()

model.save_pretrained(OUTPUT_DIR)

print("\n If there's a warning about missing keys above, please disregard :)")