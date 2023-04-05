import os
import sys
from typing import Optional, Dict, Sequence
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from dotenv import load_dotenv
load_dotenv()

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
HF_TOKEN = os.environ["HF_TOKEN"]
MICRO_BATCH_SIZE = 4
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3
LEARNING_RATE = 3e-4
CUTOFF_LEN = 1024
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
DATA_PATH = "datasets/data/diverse.jsonl"
OUTPUT_DIR = "lora-alpaca"

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    This version ensures that your embedding size is always divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    num_existing_tokens = len(tokenizer)
    num_pad_tokens = (64 - num_existing_tokens % 64) % 64

    # Add padding tokens to the tokenizer
    pad_token_dict = [{f"pad_token_{i}": f"<pad{i}>"} for i in range(num_pad_tokens)]
    tokenizer.add_tokens(pad_token_dict)

    num_total_tokens = len(tokenizer)

    model.resize_token_embeddings(num_total_tokens)

    print("Resized model...")

    if num_new_tokens > 0 or num_pad_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens - num_pad_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens - num_pad_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens - num_pad_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens - num_pad_tokens:] = output_embeddings_avg


model = LlamaForCausalLM.from_pretrained(
    "swype/deepshard-13B-raw",
    use_auth_token=HF_TOKEN,
    load_in_8bit=True,
    device_map=device_map,
)
tokenizer = LlamaTokenizer.from_pretrained(
    "swype/deepshard-13B-raw", add_eos_token=True, use_auth_token=HF_TOKEN
)

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
special_tokens = {
    "eos_token": DEFAULT_EOS_TOKEN,
    "bos_token": DEFAULT_BOS_TOKEN,
    "unk_token": DEFAULT_UNK_TOKEN,
    "pad_token": DEFAULT_PAD_TOKEN
}

smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens,
    tokenizer=tokenizer,
    model=model,
)



trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if VAL_SET_SIZE > 0 else None,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
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