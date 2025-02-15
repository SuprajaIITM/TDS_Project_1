from fastapi import FastAPI, Query, HTTPException
import os
import json
import openai
import subprocess
import sqlite3
from datetime import datetime
import shutil
import requests
import re
import base64
import numpy as np


app = FastAPI()

data_dir = "/data/"

token = os.getenv("AIPROXY_TOKEN")
if not token:
    raise Exception("AIPROXY_TOKEN is not set. Please set it in the environment variables.")

openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
openai.api_key = token
client = openai.OpenAI(api_key=token)

def classify_task(task: str):
    """Uses GPT-4o-Mini to classify a task into predefined categories using few-shot examples."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that maps tasks to predefined categories. Given a task description, return the correct category from this list: install_uv, format_md, count_weekdays, sort_contacts, extract_email, extract_credit_card_number, find_most_similar_comments, compute_gold_ticket_sales."},
                
                # Few-Shot Examples (Shows How Similar Tasks Should Be Classified)
                {"role": "user", "content": "Sort contacts in /data/contacts.json by last name and save to /data/contacts-sorted.json"},
                {"role": "assistant", "content": "sort_contacts"},
                
                {"role": "user", "content": "How many Wednesdays are there in /data/dates.txt? Save count in /data/dates-wednesdays.txt"},
                {"role": "assistant", "content": "count_weekdays"},
                
                {"role": "user", "content": "Format the file /data/format.md using Prettier 3.4.2"},
                {"role": "assistant", "content": "format_md"},
                
                {"role": "user", "content": "Find the sender’s email in /data/email.txt and save it"},
                {"role": "assistant", "content": "extract_email"},

                {
                "role": "user",
                "content": "Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt"
                },
                {
                "role": "assistant",
                "content": "extract_recent_log_lines"
                },
                {
                "role": "user",
                "content": "Find all Markdown"
                },
                {
                "role": "assistant",
                "content": "extract_markdown_titles"
                },
                {
                "role": "user",
                "content": "credit card number"
                },
                {
                "role": "assistant",
                "content": "extract_credit_card_number"
                },
                {
                "role": "user",
                "content": "Using embeddings, find the most similar pair of comments"
                },
                {
                "role": "assistant",
                "content": "find_most_similar_comments"
                },
                {
                "role": "user",
                "content": "total sales of all the items in the “Gold” ticket type?"
                },
                {
                "role": "assistant",
                "content": "compute_gold_ticket_sales"
                },
                # User Task to Classify
                {"role": "user", "content": task}
            ]
        )

        classified_task = response.choices[0].message.content.strip()
        return classified_task

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🛠️ Main Task Runner
def run_task(task: str):
    """Process and execute the given task using NLP classification."""
    try:
        classified_task = classify_task(task)  # Get structured task category

        # Map classified tasks to actual function calls
        task_mapping = {
            "install_uv": install_uv,
            #"format_md": format_md,
            #"count_weekdays": lambda: count_weekdays("Wednesday", "/data/dates.txt", "/data/dates-wednesdays.txt"),
            #"sort_contacts": sort_contacts,
            #"extract_recent_log_lines":extract_recent_log_lines,
            #"extract_markdown_titles":extract_markdown_titles,
            #"extract_email":extract_email,
            #"extract_credit_card_number": extract_credit_card_number,
            #"find_most_similar_comments":find_most_similar_comments,
            #"compute_gold_ticket_sales":compute_gold_ticket_sales
        }

        if classified_task in task_mapping:
            return task_mapping[classified_task]()  # Call corresponding function
        else:
            raise HTTPException(status_code=400, detail="Task not recognized")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def install_uv():
    """Installs uv (if missing) and runs datagen.py from URL with user email."""
    try:
        # 1️⃣ Check if 'uv' is installed, install if missing
        if shutil.which("uv") is None:
            try:
                install_proc = subprocess.run(
                    ["pip", "install", "uv"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("Installed uv:", install_proc.stdout)
            except subprocess.CalledProcessError as e:
                raise Exception("Failed to install uv: " + e.stderr)
    
        # 2️⃣ Download the datagen.py script
        datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        response = requests.get(datagen_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download datagen.py, status code: {response.status_code}")
        
        datagen_filename = "datagen.py"
        with open(datagen_filename, "w") as f:
            f.write(response.text)
        
        # 3️⃣ Ensure the local 'data/' directory exists
        local_data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(local_data_dir, exist_ok=True)  # Creates 'data/' if it doesn't exist

        # 4️⃣ Fix Windows Paths: Escape Backslashes
        local_data_dir_escaped = local_data_dir.replace("\\", "/")   # Double the backslashes
        
        # 5️⃣ Modify datagen.py to use 'data/' instead of '/data/'
        with open(datagen_filename, "r") as f:
            content = f.read()
        
        # 🔹 FIX: Replace '/data' with the properly escaped Windows path
        new_content = re.sub(r'([\'"])/data([\'"])', f'\\1{local_data_dir_escaped}\\2', content)

        # Write the modified content back to datagen.py
        with open(datagen_filename, "w") as f:
            f.write(new_content)
        
        # 6️⃣ Get the user's email from the environment
        user_email = os.getenv("USER_EMAIL")
        if not user_email:
            raise Exception("USER_EMAIL is not set. Please set it before running.")

        # 7️⃣ Run datagen.py with the user's email as the only argument
        try:
            proc = subprocess.run(
                ["python", datagen_filename, user_email],
                check=True,
                capture_output=True,
                text=True
            )
            return {"stdout": proc.stdout, "stderr": proc.stderr}
        except subprocess.CalledProcessError as e:
            raise Exception("Error running datagen.py: " + e.stderr)
    
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

@app.post("/run")
def run(task: str = Query(..., description="Task to execute")):
    """Executes the given task."""
    return run_task(task)

@app.get("/read")
def read_file(path: str = Query(..., description="Path to the file")):
    """Reads a file."""
    check_path(path)
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")