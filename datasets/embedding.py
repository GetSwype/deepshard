import openai
import os
import chromadb
import pandas as pd
import argparse

from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tqdm import tqdm
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

chroma = chromadb.Client(Settings(chroma_api_impl="rest", chroma_server_host=os.environ["CHROMA_URL"], chroma_server_http_port=80))
df = pd.read_json("./datasets/data/datasets--swype--instruct/snapshots/c504b3fb1bd408507949c50cadc83c580f4ec202/instruct.jsonl", lines=True)

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def get_embeddings(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    return [result["embedding"] for result in response["data"]]

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)

def process_collection(collection_name, text_list, metadata_list, batch_size):
    collection = chroma.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

    for i in tqdm(range(0, len(text_list), batch_size)):
        ids = [str(j) for j in range(i, i + batch_size)]
        documents = text_list[i:i + batch_size]
        embeddings = get_embeddings(text_list[i:i + batch_size])
        metadata = metadata_list[i:i + batch_size]

        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )

def main(batch_size):
    prompts = df["prompt"].tolist()
    completions = df["completion"].tolist()

    process_collection("prompts", prompts, [{"completion": completion} for completion in completions], batch_size)
    process_collection("completions", completions, [{"prompt": prompt} for prompt in prompts], batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set batch size for processing collections.")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size for processing collections.")
    args = parser.parse_args()

    main(args.batch_size)