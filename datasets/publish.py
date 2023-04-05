import random
from typing import Dict
from rq import Queue
from redis import Redis
from consume import consume_task
import psycopg2
from moderation import Celery

app = Celery('tasks', broker='pyamqp://guest@localhost//')

conn = psycopg2.connect(database="postgres",
                        host="127.0.0.1",
                        user="postgres",
                        password="passsword",
                        port="5432")

def publish_tasks(distributions: Dict[str, int], num_tasks: int, queue: any) -> None:
    task_types = []
    for task, weight in distributions.items():
        task_types += [task] * weight
    random.shuffle(task_types)

    for i in range(num_tasks):
        task_type = random.choice(task_types)
        with open(f"./prompts/{task_type}.txt", "r") as taskfile:
            prompt = taskfile.read().strip()
            task_data = {
                "task_type": task_type,
                "prompt": prompt,
            }
            queue.enqueue(consume_task, task_data)
            
def publish_moderation_tasks():
    cursor = conn.cursor()
    sql = "SELECT prompt FROM tasks_2"
    cursor.execute(sql)
    prompts = cursor.fetchall()
    print(len(prompts))

if __name__ == "__main__":
    publish_moderation_tasks()