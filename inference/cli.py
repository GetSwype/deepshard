import argparse
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, LlamaForCausalLM
from huggingface_hub import snapshot_download

# create argparser
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)


def main():
    args = parser.parse_args()
    model_name = args.model_name
    print("Loading model from: ", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
    print("Model loaded successfully\n")
    while True:
        prompt = input("Prompt: ")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        generate_gn = model.generate(inputs.input_ids, max_length=256)
        for token in generate_gn:
            print(tokenizer.decode(token, skip_special_tokens=True, clean_up_tokenization_spaces=True), end="", flush=True)
        print("\n------------------")
    


def tmp():
    args = parser.parse_args()
    model_name = args.model_name
    weights_location = snapshot_download(model_name)
    print("Loading model from: ", weights_location)
    config = AutoConfig.from_pretrained(model_name)

    with init_empty_weights(): # so that we dont have to initialize random weights
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)
        # load sharded weights (each shard = 9GB, which means 9 * 4 (bytes) * 6 shards = 216GB of GPU memory)- but this needs 80 GB A100 cuz 1 shard == 36GB + some extra files
        model = load_checkpoint_and_dispatch( 
            model, f"{weights_location}", device_map="balanced_low_0"
        )

        print("Model loaded successfully\n")
        while True:
            prompt = input("Prompt: ")
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to("cuda")
            generate_gn = model.generate(inputs.input_ids, max_length=256)
            for token in generate_gn:
                print(tokenizer.decode(token, skip_special_tokens=True, clean_up_tokenization_spaces=True), end="", flush=True)
            print("\n------------------")


if __name__ == "__main__":
    main()