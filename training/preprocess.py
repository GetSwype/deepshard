from datasets import load_dataset
from datasets import DatasetDict


# Replace 'path/to/your_file.jsonl' with the actual path to your JSONL file
dataset = load_dataset("json", data_files={"train": "/home/paperspace/Documents/deepshard/datasets/data/diverse.jsonl"}, field="data", split="train")

# Set the proportion of data you want in your training set (e.g., 0.8 for 80%)
train_proportion = 0.8
num_train_samples = int(len(dataset) * train_proportion)

# Shuffle the dataset
dataset = dataset.shuffle()

# Split the dataset into train and test sets
train_set = dataset.select(range(num_train_samples))
test_set = dataset.select(range(num_train_samples, len(dataset)))

# Create a DatasetDict containing the train and test sets
dataset_dict = DatasetDict({"train": train_set, "test": test_set})
# Replace 'path/to/save/directory' with the actual path to the directory where you want to save the files
dataset_dict.save_to_disk("/home/paperspace/Documents/deepshard/datasets/data/processed/")