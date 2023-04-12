import json
import openai
import os

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed


# Load environment variables
load_dotenv()

# Load OpenAI API key and PostgreSQL URL
api_key = os.environ["OPENAI_API_KEY"]

openai.api_key = api_key

# Call OpenAI API with retries
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai_api(prompt, is_prompt):
    system_message = "You are a helpful agent" if is_prompt else "You are an intelligent agent who thinks carefully about what to say"
    new_prompt = "Re-write the following prompt outputting nothing except the re-written prompt. DO NOT BEGIN WITH 'Rewritten Prompt: '! Here is your prompt: " + prompt if is_prompt else prompt
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
        completion = response.choices[0].message['content'].strip().replace("Rewritten prompt: ", "")
    return completion


# Process a single row of data
def process_prompt(request):
    request_json = request.get_json(silent=True)
    prompt = request_json['prompt']

    try:
        rewritten_prompt = call_openai_api(prompt, True)
        completion = call_openai_api(rewritten_prompt, False)
    except Exception as e:
        print(f"Error processing prompt: {prompt}, error: {e}")

    return json.dumps({"rewritten_prompt": rewritten_prompt, "completion": completion}), 200