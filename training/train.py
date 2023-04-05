from dotenv import load_dotenv
load_dotenv()

import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import logging
from transformers import logging as hf_logging
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
import evaluate

hf_logging.set_verbosity(logging.DEBUG)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
HF_TOKEN = os.environ["HF_TOKEN"]

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class ExtraArgs:
    search: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    fp16: bool = field(
        default=True
    )
    fsdp_transformer_layer_cls_to_wrap: Optional[str] = field(
        default=None,
        metadata={"help": "The FSDP transformer layer class to wrap"}
    )
    fsdp: Optional[str] = field(
        default="full_shard auto_wrap",
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


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




def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
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
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # Read the JSONL file using pandas
        data = pd.read_json(data_path, lines=True)
        data = data.sample(frac=1).reset_index(drop=True)

        prompts = data["prompt"].tolist()
        completions = data["completion"].tolist()

        logging.warning("Formatting inputs...")
        
        sources = [
            f'You are a helpful, obedient and informative instruction-following AI. Create a response to the instruction below:\nInstruction: {source}\nResponse:'
            for source in prompts
        ]
        targets = [target for target in completions]

        print("Example source: ", sources[0])
        print("Example target: ", targets[0])

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
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

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_path)

    print("Dataset size: ", len(train_dataset))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ExtraArgs))
    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

    print("Training Args: ", training_args)

    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_auth_token=HF_TOKEN
    )

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_auth_token=HF_TOKEN,
        use_fast=False,
    )

    tokenizer.bos_token = DEFAULT_BOS_TOKEN
    tokenizer.eos_token = DEFAULT_EOS_TOKEN
    tokenizer.unk_token = DEFAULT_UNK_TOKEN
    tokenizer.add_eos_token = True

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

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    if extra_args.search:
        def model_init():
            return transformers.LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                use_auth_token=HF_TOKEN,
                return_dict=True,
            )
        training_args = TrainingArguments(
            "test", evaluation_strategy="steps", eval_steps=100, do_eval=True)
        trainer = Trainer(tokenizer=tokenizer, args=training_args, **data_module, compute_metrics=compute_metrics, model_init=model_init)
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="ray",
            resources_per_trial={"gpu": 8},
            # Choose among many libraries:
            # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
            search_alg=HyperOptSearch(metric="objective", mode="max"),
            # Choose among schedulers:
            # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
            scheduler=ASHAScheduler(metric="objective", mode="max"))
        print("Best trial: ", best_trial)
    else:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        trainer.train()
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()