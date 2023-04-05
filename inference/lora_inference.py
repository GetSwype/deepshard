import os
import torch

from transformers import LlamaForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from dotenv import load_dotenv

load_dotenv()

device = torch.device("cuda:0")  # Change this to your desired device

peft_model_id = "/home/ubuntu/deepshard/lora-alpaca"
config = PeftConfig.from_pretrained(peft_model_id)
model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path, use_auth_token=os.environ.get("HF_TOKEN"))
model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_auth_token=os.environ.get("HF_TOKEN"))

# model = model.to(device)
# model.eval()

source = "Write a tweet by donald trump"
ipt = f'<s>Follow all instructions and respond truthfully and accurately.\nInstructions: {source}\nResponse:'
inputs = tokenizer(ipt, return_tensors="pt")

# Move input tensors to the same device as the model
# inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=256)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
