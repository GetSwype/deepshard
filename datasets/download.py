import pandas as pd

from dotenv import load_dotenv
load_dotenv()

import os
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

# Replace these values with the appropriate information
dataset_name = "swype/instruct-66k"
HF_TOKEN = os.environ['HF_TOKEN']
file_path = "diverse.jsonl"  # Replace with the path of the file in the repo
destination_dir = "datasets/data"  # Replace with the path to the local directory where you want to save the file

file_spot = hf_hub_download(
    repo_id=dataset_name,
    filename=file_path,
    token=HF_TOKEN,
    repo_type="dataset",
    cache_dir=destination_dir,
)

# Read the data
main = pd.read_json(file_spot, lines=True)
secondary = pd.read_json("/root/documents/deepshard/datasets/data.jsonl", lines=True)
secondary.rename(columns={'response': 'completion'}, inplace=True)
secondary.drop('source', axis=1, inplace=True)
merged_df = pd.merge(main, secondary, how='outer')

# Split the data into train and test sets
train, test = train_test_split(merged_df, test_size=0.2, random_state=42)

train = train[:102_400]
test = test[:1024]

# Save the train and test sets as JSONL files
train.to_json("datasets/data/train.jsonl", orient="records", lines=True)
test.to_json("datasets/data/test.jsonl", orient="records", lines=True)