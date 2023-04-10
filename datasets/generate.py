import json
import pandas as pd
import openai
import os
import psycopg2

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# Load environment variables
load_dotenv()

# Load OpenAI API key and PostgreSQL URL
api_key = os.environ["OPENAI_API_KEY"]
postgres_url = os.environ["POSTGRES_URL"]

openai.api_key = api_key

# Read JSONL file
def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Connect to PostgreSQL database
def connect_to_db():
    return psycopg2.connect(postgres_url)

# Insert data into the PostgreSQL table
def insert_data(prompt, completion):
    conn = connect_to_db()
    cursor = conn.cursor()
    query = "INSERT INTO data (prompt, completion) VALUES (%s, %s);"
    cursor.execute(query, (prompt, completion))
    conn.commit()
    cursor.close()
    conn.close()

# Call OpenAI API with retries
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai_api(prompt, is_prompt):
    system_message = "You are tasked with re-writing a prompt. You will be given a prompt that you must re-write." if is_prompt else "You are an intelligent agent with thoughts and feelings."
    new_prompt = prompt if is_prompt else open("datasets/prompts/thought.txt").read()+prompt
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": new_prompt}
        ],
        max_tokens=1024,
        n=1,
        temperature=0.5,
    )
    completion = response.choices[0].message['content'].strip()
    if is_prompt:
        return completion
    else:
        try:
            obj = json.loads(completion)
            obj["thought"]
            obj["completion"]
            return completion
        except:
            return {"thought": "", "completion": ""}


# Process a single row of data
def process_row(row):
    prompt = row["prompt"]
    try:
        rewritten_prompt = call_openai_api(prompt, True)
        completion = call_openai_api(rewritten_prompt, False)
        insert_data(rewritten_prompt, completion)
    except Exception as e:
        print(f"Error processing prompt: {prompt}, error: {e}")
        completion = None
        rewritten_prompt = None

    row["next_gen_completion"] = completion
    row["rewritten_prompt"] = rewritten_prompt
    return row

# Main function
def main(file_path):
    json_data = read_jsonl(file_path)
    df = pd.DataFrame(json_data)

    # Wrap the iterable with tqdm for a progress bar
    progress_bar = tqdm([row for _, row in df.iterrows()])

    # Process prompts using multiprocessing
    with Pool(cpu_count()) as pool:
        processed_data = pool.map(process_row, progress_bar)

    # Update dataframe and save to a new JSONL file
    df = pd.DataFrame(processed_data)
    df.to_json("output.jsonl", orient="records", lines=True)

if __name__ == "__main__":
    file_path = "datasets/data/datasets--swype--instruct/snapshots/c504b3fb1bd408507949c50cadc83c580f4ec202/instruct.jsonl"
    main(file_path)
