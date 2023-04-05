import argparse
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

import os
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

# Set up the command line arguments
parser = argparse.ArgumentParser(description="Process dataset with max_train and max_test")
parser.add_argument("--max_train", type=int, help="maximum number of train items")
parser.add_argument("--max_test", type=int, help="maximum number of test items")
args = parser.parse_args()

# Replace these values with the appropriate information
dataset_name = "swype/instruct"
HF_TOKEN = os.environ['HF_TOKEN']
file_path = "instruct.jsonl"  # Replace with the path of the file in the repo
destination_dir = "datasets/data"  # Replace with the path to the local directory where you want to save the file

file_spot = hf_hub_download(
    repo_id=dataset_name,
    filename=file_path,
    token=HF_TOKEN,
    repo_type="dataset",
    cache_dir=destination_dir,
)

def process_data(max_train=None, max_test=None):
    # Read the data
    main = pd.read_json(file_spot, lines=True)

    # Split the data into train and test sets
    train, test = train_test_split(main, test_size=0.2, random_state=42)

    # Set default values for max_train and max_test if not provided
    max_train = max_train or len(train)
    max_test = max_test or len(test)

    # Ensure max_train and max_test are not greater than the respective split sizes
    max_train = min(max_train, len(train))
    max_test = min(max_test, len(test))

    # Slice the data using max_train and max_test
    train = train[:max_train]
    test = test[:max_test]

    # Save the train and test sets as JSONL files
    train.to_json("datasets/data/train.jsonl", orient="records", lines=True)
    test.to_json("datasets/data/test.jsonl", orient="records", lines=True)

# Call the function with the command line arguments
process_data(args.max_train, args.max_test)