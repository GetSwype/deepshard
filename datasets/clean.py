import openai
import os
import psycopg2
import random

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

# Function to clean prompt and completion strings
def clean_string(s):
    # Remove leading and trailing spaces, including non-breaking spaces
    s = s.strip()

    # Replace ".:" with either ":" or ". " randomly
    while ".:" in s:
        s = s.replace(".:", random.choice([":", ". "]), 1)

    # Remove double quotes at the start and end of the string
    s = s.strip('"')

    return s



# Function to process and update data in batches
def process_data_in_batches(batch_size=500):
    offset = 0

    while True:
        cur.execute(
            "SELECT id, prompt, completion FROM tasks_2 LIMIT %s OFFSET %s",
            (batch_size, offset),
        )
        rows = cur.fetchall()

        if not rows:
            break

        for row in rows:
            id, prompt, completion = row
            cleaned_prompt = clean_string(prompt)
            cleaned_completion = clean_string(completion)

            cur.execute(
                "UPDATE tasks_2 SET prompt = %s, completion = %s WHERE id = %s",
                (cleaned_prompt, cleaned_completion, id),
            )

        conn.commit()
        offset += batch_size

    print("Data processing and updating completed.")


# Call the function to process data in batches
process_data_in_batches()

# Close cursor and connection
cur.close()
conn.close()

