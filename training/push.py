from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()
api = HfApi()

HF_TOKEN = os.environ["HF_TOKEN"]

api.upload_folder(
    folder_path="/home/paperspace/Documents/deepshard/finetuned/",
    repo_id="swype/deepshard-13B-ft",
    repo_type="model",
    use_auth_token=HF_TOKEN
)