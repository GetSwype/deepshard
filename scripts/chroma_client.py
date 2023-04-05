import chromadb
from chromadb.config import Settings
import psycopg2

from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )
chroma = chromadb.Client(Settings(chroma_api_impl="rest",
                                  chroma_server_host="localhost",
                                  chroma_server_http_port=8000))

collection = chroma.get_or_create_collection(name="deepshard_collection_4", embedding_function=openai_ef)

answer = collection.query(query_texts=["Write a python script"])

print(answer)
