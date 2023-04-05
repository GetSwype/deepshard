import openai
import os
import psycopg2
import openai
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


openai.api_key = os.environ['OPENAI_API_KEY']

conn = psycopg2.connect(database="postgres",
                        host="127.0.0.1",
                        user="postgres",
                        password="passsword",
                        port="5432")


def main():
    cur = conn.cursor()
    cur.execute("SELECT id, prompt FROM tasks_2")
    rows = cur.fetchall()
    # convert rows list to chunks of 500
    batches = [rows[i:i + 500] for i in range(0, len(rows), 500)]

    for batch in tqdm(batches):
        # create embedding
        response = openai.Embedding.create(
            input=[row[1] for row in batch],
            model="text-embedding-ada-002"
        )
        response = response.to_dict()
        # chunk updates postgres with embedding
        for result in response["data"]:
            id = batch[result["index"]][0]
            cur.execute("UPDATE tasks_2 SET embedding=%s WHERE id=%s", (result["embedding"], id))
        conn.commit()
    

if __name__ == "__main__":
    main()