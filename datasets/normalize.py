import openai
import json
import random
import psycopg2
from tenacity import retry, wait_exponential, stop_never

@retry(wait=wait_exponential(), stop=stop_never())
def generate_completion(prompt):
    random_temp = round(random.uniform(0.2, 0.8), 2)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", max_tokens=256, n=1, temperature=random_temp, messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt
        }
    ])
    completion = response.choices[0].message.content.strip()
    return completion

conn = psycopg2.connect(database="postgres",
                        host="127.0.0.1",
                        user="postgres",
                        password="passsword",
                        port="5432")

with open("alpaca.json", "r") as f:
    data = json.load(f)
    # For each entry in the json array, take the "instruction" key, and "input" key. If the instruction has no input, the prompt is just the instruction. If the instruction has input, the prompt is the instruction + the input.
    for entry in data:
        prompt = entry["instruction"]
        if entry["input"] != "":
            # 50 % of the time, put a new line before the input
            if random.random() > 0.5:
                prompt += ":\n" + entry["input"]
            else:
                prompt += ": " + entry["input"]
        # Then, we can use the prompt to generate a completion. We can then write the completion to a file.
        completion = generate_completion(prompt)
        # You have prompt
        task_data = {
            "task_type": "generation",
            "prompt": prompt,
            "completion": completion,
        }
        cursor = conn.cursor()
        sql = """INSERT INTO data(task_type, prompt, completion)
                VALUES(%s, %s, %s)"""
        cursor.execute(sql, (task_data["task_type"], task_data["prompt"], task_data["completion"]))
        conn.commit()
