import os
import json
import psycopg2

from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    database="postgres",
    host=os.environ['RDS_HOST'],
    user=os.environ['RDS_USERNAME'],
    password=os.environ['RDS_PASSWORD'],
    port="5432",
)

cur = conn.cursor()

# Function to fetch data from the database
def fetch_data():
    cur.execute("SELECT prompt, completion FROM tasks_2")
    rows = cur.fetchall()
    return rows

# Function to write data to a JSON Lines file
def write_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for row in data:
            prompt, completion = row
            json_data = {"prompt": prompt, "completion": completion}
            f.write(json.dumps(json_data) + '\n')

data = fetch_data()
write_to_jsonl(data, 'diverse.jsonl')

# Close cursor and connection
cur.close()
conn.close()