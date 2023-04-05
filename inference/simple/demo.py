import argparse
import gc
import math
import os
import time

import torch
import torch.distributed as dist

from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float16")

    return parser.parse_args()


t_start = time.time()

num_tokens = 256

args = get_args()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = torch.cuda.device_count()

rank = local_rank


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)

print_rank0(f"Using {world_size} gpus")
model_name = "swype/deepshard-13B-ft"
print_rank0(f"Loading model {model_name}")

tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)

# XXX: can't automatically derive dtype via config's `from_pretrained`
dtype = torch.float16

# print(get_max_memory_per_gpu_dict())

infer_dtype = args.dtype
if infer_dtype == "int8":
    dtype = torch.int8

kwargs = dict(
    device_map="auto",
    use_auth_token=HF_TOKEN,
)


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


# balanced_low_0 - because it allows a larger batch size with multiple GPUs
if get_world_size() > 1:
    kwargs["device_map"] = "balanced_low_0"


if infer_dtype == "int8":
    print_rank0("Using `load_in_8bit=True` to use quanitized model")
    kwargs["load_in_8bit"] = True
else:
    kwargs["torch_dtype"] = dtype


model = LlamaForCausalLM.from_pretrained(model_name, **kwargs)


if args.benchmark:
    t_ready = time.time()


### Generate

print_rank0(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")


def generate(inputs):
    """returns a list of zipped inputs, outputs and number of new tokens"""

    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda:0")

    generate_kwargs["eos_token_id"] = tokenizer.encode("</s>")[0]
    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)


ipt = input("Input: ")

while True:

    generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=True, temperature=0.6, repetition_penalty=1.1)
    # generate_kwargs = dict(max_new_tokens=num_tokens, use_cache=False, do_sample=False)
    # generate_kwargs = dict(min_length=num_tokens, max_length=num_tokens, do_sample=False)

    inputs = [f"You are a friendly and helpful AI model capable of responding to any prompt accurately and concisely.\n\nPrompt: {ipt}\nResponse:"]
    t_generate_start = time.time()
    generated = generate(inputs)
    t_generate_span = time.time() - t_generate_start
    for i, o, _ in generated:
        print(o.replace(inputs[0], ""))

    ipt = input("\nInput: ")
