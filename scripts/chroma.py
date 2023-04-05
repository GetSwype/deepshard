import chromadb
from chromadb.config import Settings
import psycopg2
from tqdm import tqdm
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )
chroma = chromadb.Client(Settings(chroma_api_impl="rest",
                                  chroma_server_host="localhost",
                                  chroma_server_http_port=8000))

# Connect to the database
conn = psycopg2.connect(
    host=RDS_HOST,
    database=RDS_DB,
    user=RDS_USER,
    password=RDS_PASSWORD
)

collection = chroma.get_or_create_collection(name="deepshard_sft_2", embedding_function=openai_ef)
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM tasks_2;")
total_rows = cur.fetchone()[0]


# Set the batch size
batch_size = 2500

# Loop through the rows
for offset in tqdm(range(0, total_rows, batch_size)):
    # Get the next batch of rows
    cur.execute(f"SELECT id, prompt, embedding, completion FROM tasks_2 ORDER BY id OFFSET {offset} LIMIT {batch_size}")
    rows = cur.fetchall()
    
    ids = []
    documents = []
    embeddings = []
    metadata = []
    for row in rows:
        ids.append(str(row[0]))
        documents.append(row[1])
        embeddings.append(row[2])
        metadata.append({"completion": row[3]})
    
    # print(ids)
    # print()
    # # print(documents)
    # print()
    # print(embeddings)
    # break
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadata,
        ids=ids
    )


# Close the cursor and connection
cur.close()
conn.close()
