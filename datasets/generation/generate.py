import json
import requests
import psycopg2
import os

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


load_dotenv()

# Load OpenAI API key and PostgreSQL URL
postgres_url = os.environ["POSTGRES_URL"]

# Connect to PostgreSQL database
def connect_to_db():
    return psycopg2.connect(postgres_url)

# Insert data into the PostgreSQL table
def insert_data(prompt, completion):
    conn = connect_to_db()
    cursor = conn.cursor()
    query = "INSERT INTO dataset (prompt, completion) VALUES (%s, %s);"
    # Convert the dictionary to a JSON string
    cursor.execute(query, (prompt, completion))
    conn.commit()
    cursor.close()
    conn.close()

def call_cloud_function(prompt):
    url = "https://us-central1-norse-avatar-379521.cloudfunctions.net/process_prompt"
    data = {"prompt": prompt}
    response = requests.post(url, json=data)
    return response.json()

def process_dataset(file_path, batch_size=600):
    # Read JSONL file
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))

    # Call the Cloud Function in parallel
    with ThreadPoolExecutor() as executor:
        for i in tqdm(range(0, len(data), batch_size), desc="Processing"):
            batch = data[i:i+batch_size]
            futures = {executor.submit(call_cloud_function, row["prompt"]): row["prompt"] for row in batch}

            for future in as_completed(futures):
                result = future.result()
                insert_data(result["rewritten_prompt"], result["completion"])

if __name__ == "__main__":
    file_path = "datasets/data/datasets--swype--instruct/snapshots/c504b3fb1bd408507949c50cadc83c580f4ec202/instruct.jsonl"
    process_dataset(file_path)
