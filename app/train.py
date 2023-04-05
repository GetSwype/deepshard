import copy
import os
from typing import Dict, Sequence
import pandas as pd
import gradio as gr
import random
import transformers
import torch
import logging
import datasets
import gc
import peft
from transformers import logging as hf_logging
from sklearn.model_selection import train_test_split
from transformers import Trainer
from torch.utils.data import Dataset
import evaluate



hf_logging.set_verbosity(logging.DEBUG)

model = None
tokenizer = None
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
HF_TOKEN = os.environ["HF_TOKEN"]

def load_model(model_name):
    global model
    print('Loading base model...')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

def load_tokenizer(model_name):
    global tokenizer
    print('Loading tokenizer...')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, padding_side="right", use_fast=False
    )
    tokenizer.bos_token = DEFAULT_BOS_TOKEN
    tokenizer.eos_token = DEFAULT_EOS_TOKEN
    tokenizer.unk_token = DEFAULT_UNK_TOKEN
    tokenizer.add_eos_token = True

def reset_model():
    global model
    global tokenizer

    del model
    del tokenizer

    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    model = None
    tokenizer = None


def parse_file(file):
    print(file)
    df = pd.read_json(file, lines=True)
    print(df.head())

    if (df.empty):
        raise gr.Error("File is empty")
    if df.get("prompt", None) is None:
        raise gr.Error("File does not contain a prompt column")
    if df.get("completion", None) is None:
        raise gr.Error("File does not contain a completion column")
    if (df["prompt"].isnull().values.any()):
        raise gr.Error("File contains null prompts")
    if (df["completion"].isnull().values.any()):
        raise gr.Error("File contains null completions")

    # split df into train and eval
    train, eval = train_test_split(df, test_size=0.2)
    return train, eval

    

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, train_df, eval_df) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_frame=train_df)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_frame=eval_df)

    print("Dataset size: ", len(train_dataset))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)



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
    # pad_token_dict = [{f"pad_token_{i}": f"<pad{i}>"} for i in range(num_pad_tokens)]
    # tokenizer.add_tokens(pad_token_dict)

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



class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(
        self,
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [self._tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def __init__(self, data_frame: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        # Read the JSONL file using pandas
        data = data_frame.sample(frac=1).reset_index(drop=True)

        prompts = data["prompt"].tolist()
        completions = data["completion"].tolist()

        logging.warning("Formatting inputs...")
        
        sources = [
            source
            for source in prompts
        ]
        targets = [target for target in completions]

        print("Example source: ", sources[0])
        print("Example target: ", targets[0])

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def tokenize_and_train(
    training_file,
    model_type,
    max_seq_length,
    micro_batch_size,
    gradient_accumulation_steps,
    epochs,
    learning_rate,
    model_name,
):
    progress = gr.Progress(track_tqdm=True)
    global model
    global tokenizer

    if model_type is None:
        raise gr.Error("No model is selected")
    if (training_file is None):
        raise gr.Error("No training file provided")
    if (model_name is None):
        raise gr.Error("No model name provided")


    if (model is None): load_model(model_type)
    if (tokenizer is None): load_tokenizer(model_type)

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

    train_df, eval_df = parse_file(training_file.name)
    data_module = make_supervised_data_module(tokenizer=tokenizer, train_df=train_df, eval_df=eval_df)
    training_args = transformers.TrainingArguments(
        # Set the batch size for training on each device (GPU, CPU, or TPU).
        per_device_train_batch_size=micro_batch_size, 

        # Number of steps for gradient accumulation. This is useful when the total 
        # batch size is too large to fit in GPU memory. The effective batch size 
        # will be the product of 'per_device_train_batch_size' and 'gradient_accumulation_steps'.
        gradient_accumulation_steps=gradient_accumulation_steps,  

        # Number of warmup steps for the learning rate scheduler. During these steps, 
        # the learning rate increases linearly from 0 to its initial value. Warmup helps
        #  to reduce the risk of very large gradients at the beginning of training, 
        # which could destabilize the model.
        # warmup_steps=100, 

        # The total number of training steps. The training process will end once this 
        # number is reached, even if not all the training epochs are completed.
        # max_steps=1500, 

        # The total number of epochs (complete passes through the training data) 
        # to perform during the training process.
        num_train_epochs=epochs,  

        # The initial learning rate to be used during training.
        learning_rate=learning_rate, 

        # Enables mixed precision training using 16-bit floating point numbers (FP16). 
        # This can speed up training and reduce GPU memory consumption without 
        # sacrificing too much model accuracy.
        fp16=True,  

        # The frequency (in terms of steps) of logging training metrics and statistics 
        # like loss, learning rate, etc. In this case, it logs after every 20 steps.
        logging_steps=20, 

        # The output directory where the trained model, checkpoints, 
        # and other training artifacts will be saved.
        output_dir=f"models/{model_name}-out", 

        # The maximum number of checkpoints to keep. When this limit is reached, 
        # the oldest checkpoint will be deleted to save a new one. In this case, 
        # a maximum of 3 checkpoints will be kept.
        save_total_limit=3,  
    )

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    result = trainer.train()
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # model.config.use_cache = False
    # result = trainer.train(resume_from_checkpoint=False)
    # model.save_pretrained(output_dir)

    del data
    reset_model()

    return result

def random_hyphenated_word():
    word_list = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig']
    word1 = random.choice(word_list)
    word2 = random.choice(word_list)
    return word1 + '-' + word2

def training_tab():
    with gr.Tab("Finetuning"):

        with gr.Column():
            training_file = gr.File(label="Training Data File", info="Must be a jsonl document with prompt and completion", file_types=[".jsonl"])


        with gr.Row():
            with gr.Column():
                max_seq_length = gr.Slider(
                    minimum=1, maximum=4096, value=512,
                    label="Max Sequence Length", 
                    info="The maximum length of each sample text sequence. Sequences longer than this will be truncated."
                )

                micro_batch_size = gr.Slider(
                    minimum=1, maximum=100, value=1, 
                    label="Micro Batch Size", 
                    info="The number of examples in each mini-batch for gradient computation. A smaller micro_batch_size reduces memory usage but may increase training time."
                )

                gradient_accumulation_steps = gr.Slider(
                    minimum=1, maximum=10, value=1, 
                    label="Gradient Accumulation Steps", 
                    info="The number of steps to accumulate gradients before updating model parameters. This can be used to simulate a larger effective batch size without increasing memory usage."
                )

                epochs = gr.Slider(
                    minimum=1, maximum=100, value=1, 
                    label="Epochs",
                    info="The number of times to iterate over the entire training dataset. A larger number of epochs may improve model performance but also increase the risk of overfitting.")

                learning_rate = gr.Slider(
                    minimum=0.00001, maximum=0.01, value=3e-4,
                    label="Learning Rate",
                    info="The initial learning rate for the optimizer. A higher learning rate may speed up convergence but also cause instability or divergence. A lower learning rate may require more steps to reach optimal performance but also avoid overshooting or oscillating around local minima."
                )

            with gr.Column():
                with gr.Column():
                    model_type = gr.Dropdown(
                        ["llama-7B", "llama-13B", "llama-30B", "bigscience/bloom-560m"], label="Model type", info="Kind of model to use for training"
                    )

                    model_name = gr.Textbox(
                        lines=1, label="What do you want to name this model?", value=random_hyphenated_word()
                    )

                    with gr.Row():
                        train_btn = gr.Button(
                            "Train", variant="primary", label="Train", 
                        )

                        abort_button = gr.Button(
                            "Abort", label="Abort", 
                        )
    
        output_text = gr.Text("Training Status")

        train_progress = train_btn.click(
            fn=tokenize_and_train,
            inputs=[
                training_file,
                model_type,
                max_seq_length,
                micro_batch_size,
                gradient_accumulation_steps,
                epochs,
                learning_rate,
                model_name
            ],
            outputs=output_text
        )

        abort_button.click(None, None, None, cancels=[train_progress])