from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import PlainTextResponse
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
from dateutil import parser
import itertools


app = FastAPI()

data_dir = "/data/"

token = os.getenv("AIPROXY_TOKEN")
print(token)
if not token:
    raise Exception("AIPROXY_TOKEN is not set. Please set it in the environment variables.")

openai.api_base = "http://aiproxy.sanand.workers.dev/openai/v1"
openai.api_key = token

def classify_task(task: str):
    """Uses GPT-4o-Mini to classify a task into predefined categories using few-shot examples."""
    try:
        response = openai.ChatCompletion.create(
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
        #print("🔹 Raw Response:", response) 
        classified_task = response.choices[0].message.content.strip()
        return classified_task

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def compute_gold_ticket_sales():
    try:
        # 🔹 Define file paths
        data_dir = os.path.join(os.getcwd(), "data")
        db_path = os.path.join(data_dir, "ticket-sales.db")
        output_file = os.path.join(data_dir, "ticket-sales-gold.txt")

        # 🔹 Ensure the database file exists
        if not os.path.exists(db_path):
            raise Exception(f"Database file not found: {db_path}")

        # 🔹 Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 🔹 Query total sales for "Gold" ticket type
        cursor.execute("""
            SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'
        """)
        result = cursor.fetchone()[0]  # Fetch first column of first row

        conn.close()

        # 🔹 Handle case where no Gold tickets exist
        total_sales = result if result is not None else 0

        # 🔹 Write total sales to output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(str(total_sales))

        return {"status": "success", "message": f"Total sales for 'Gold' tickets saved to {output_file}", "total_sales": total_sales}

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
            "format_md": format_md,
            "count_weekdays": lambda: count_weekdays("Wednesday", "/data/dates.txt", "/data/dates-wednesdays.txt"),
            "sort_contacts": sort_contacts,
            "extract_recent_log_lines":extract_recent_log_lines,
            "extract_markdown_titles":extract_markdown_titles,
            "extract_email":extract_email,
            "extract_credit_card_number": extract_credit_card_number,
            "find_most_similar_comments":find_most_similar_comments,
            "compute_gold_ticket_sales":compute_gold_ticket_sales
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

def format_md():
    try:
        # Define the local data directory (project-root/data)
        local_data_dir = os.path.join(os.getcwd(), "data")

        # Construct the local file path for format.md
        file_path = os.path.join(local_data_dir, "format.md")

        # Check if the file exists
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")

        # Find full paths for `npx` and `prettier`
        npm_path = shutil.which("npm")
        npx_path = shutil.which("npx") or "C:\\Program Files\\nodejs\\npx.cmd"  # Set manually if needed
        prettier_path = shutil.which("prettier")

        # If Prettier is globally installed, use it
        if prettier_path:
            prettier_cmd = [prettier_path, "--write", file_path]
        elif npx_path:
            prettier_cmd = [npx_path, "prettier@3.4.2", "--write", file_path]
        else:
            raise Exception("Prettier and npx not found. Please install Node.js and Prettier.")

        # Run Prettier to format the file in-place
        proc = subprocess.run(
            prettier_cmd,
            check=True,
            capture_output=True,
            text=True
        )

        return {"status": "success", "message": "Markdown file formatted", "stdout": proc.stdout, "stderr": proc.stderr}

    except subprocess.CalledProcessError as e:
        raise Exception("Error running Prettier: " + e.stderr)
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

def count_weekdays(weekday, input_file, output_file):
    try:
        local_data_dir = os.path.join(os.getcwd(), "data")

        input_path = os.path.join(local_data_dir, os.path.basename(input_file))
        output_path = os.path.join(local_data_dir, os.path.basename(output_file))

        if not os.path.exists(input_path):
            raise Exception(f"File not found: {input_path}")

        with open(input_path, "r") as f:
            raw_dates = [line.strip() for line in f.readlines()]

        count = 0
        incorrect_parses = 0

        for date_str in raw_dates:
            if not date_str:
                continue  # Skip empty lines

            parsed_date = None

            # 🔹 First, try parsing with known formats explicitly
            for fmt in [
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d-%b-%Y",   # 07-Mar-2011
                "%d-%B-%Y",   # 07-March-2011
                "%b %d, %Y",  # Aug 27, 2011
                "%B %d, %Y",  # August 27, 2011
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S"
            ]:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    break  # Exit if successfully parsed
                except ValueError:
                    continue  # Try next format

            # 🔹 If no match found, try automatic parsing (as a fallback)
            if not parsed_date:
                try:
                    parsed_date = parser.parse(date_str, dayfirst=True, fuzzy=False)
                except Exception:
                    incorrect_parses += 1
                    print(f"⚠️ Failed to parse: {date_str}")
                    continue  # Skip this entry

            # 🔹 Ensure we only count correctly parsed dates
            if parsed_date.strftime("%A") == weekday:
                print(parsed_date.weekday())
                count += 1
            
        # 🔹 Write the count to the output file
        with open(output_path, "w") as f:
            f.write(str(count))

        return {
            "status": "success",
            "message": f"{weekday} count written to {output_path}",
            "count": count,
            "failed_parses": incorrect_parses
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def sort_contacts():
    try:
        # 🔹 Define the local `data/` directory
        local_data_dir = os.path.join(os.getcwd(), "data")

        # 🔹 Construct file paths
        input_path = os.path.join(local_data_dir, "contacts.json")
        output_path = os.path.join(local_data_dir, "contacts-sorted.json")

        # 🔹 Ensure the input file exists
        if not os.path.exists(input_path):
            raise Exception(f"File not found: {input_path}")

        # 🔹 Read contacts from the JSON file
        with open(input_path, "r") as f:
            contacts = json.load(f)

        # 🔹 Sort contacts by last name, then first name
        contacts.sort(key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))

        # 🔹 Write the sorted contacts to a new file
        with open(output_path, "w") as f:
            json.dump(contacts, f, indent=2)

        return {"status": "success", "message": "Contacts sorted", "output_file": output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_recent_log_lines():
    try:
        # 🔹 Define local `logs/` directory
        logs_dir = os.path.join(os.getcwd(), "data", "logs")
        output_file = os.path.join(os.getcwd(), "data", "logs-recent.txt")

        # 🔹 Ensure the `logs/` directory exists
        if not os.path.exists(logs_dir):
            raise Exception(f"Logs directory not found: {logs_dir}")

        # 🔹 Get all `.log` files in `logs/`, sorted by modification time (newest first)
        log_files = [
            os.path.join(logs_dir, f)
            for f in os.listdir(logs_dir) if f.endswith(".log")
        ]
        log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # Sort by modified time

        # 🔹 Extract first lines from the 10 most recent `.log` files
        first_lines = []
        for log_file in log_files[:10]:  # Limit to 10 most recent logs
            with open(log_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()  # Read first line
                if first_line:
                    first_lines.append(first_line)

        # 🔹 Write to `logs-recent.txt`
        with open(output_file, "w", encoding="utf-8") as f:
            for line in first_lines:
                f.write(line + "\n")

        return {"status": "success", "message": f"Extracted first lines from {len(first_lines)} logs.", "output_file": output_file}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_email():
    try:
        # 🔹 Define file paths
        data_dir = os.path.join(os.getcwd(), "data")
        input_file = os.path.join(data_dir, "email.txt")
        output_file = os.path.join(data_dir, "email-sender.txt")

        # 🔹 Ensure input file exists
        if not os.path.exists(input_file):
            raise Exception(f"File not found: {input_file}")

        # 🔹 Read the email content
        with open(input_file, "r", encoding="utf-8") as f:
            email_content = f.read()

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract only the sender's email address from the given email content."},
                {"role": "user", "content": email_content}
            ]
        )

        sender_email = response.choices[0].message.content.strip()

        # 🔹 Write extracted email to output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(sender_email)

        return {"status": "success", "message": f"Sender email extracted and saved to {output_file}", "email": sender_email}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_credit_card_number():
    try:
        # 🔹 Define file paths
        data_dir = os.path.join(os.getcwd(), "data")
        input_file = os.path.join(data_dir, "credit_card.png")
        output_file = os.path.join(data_dir, "credit-card.txt")

        # 🔹 Ensure input file exists
        if not os.path.exists(input_file):
            raise Exception(f"File not found: {input_file}")

        
        # 🔹 Read the image in binary mode
        with open(input_file, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract only the number from the square box given image."},
                {"role": "user", "content": "Here's the image of a square. Extract number", "image": {"data": base64_image, "mime_type": "image/png"}}
            ]
        )

        # 🔹 Extract and clean the credit card number
        card_number = response.choices[0].message.content.strip().replace(" ", "")

        # 🔹 Write extracted number to output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(card_number)

        return {"status": "success", "message": f"Credit card number extracted and saved to {output_file}", "card_number": card_number}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_most_similar_comments():
    try:
        # 🔹 Define file paths
        data_dir = os.path.join(os.getcwd(), "data")
        input_file = os.path.join(data_dir, "comments.txt")
        output_file = os.path.join(data_dir, "comments-similar.txt")

        # 🔹 Ensure input file exists
        if not os.path.exists(input_file):
            raise Exception(f"File not found: {input_file}")

        # 🔹 Read comments from file
        with open(input_file, "r", encoding="utf-8") as f:
            comments = [line.strip() for line in f.readlines() if line.strip()]

        if len(comments) < 2:
            raise Exception("Not enough comments to compare.")

        # 🔹 Initialize OpenAI API with AIProxy
        token = os.getenv("AIPROXY_TOKEN")
        if not token:
            raise Exception("❌ AIPROXY_TOKEN is NOT set! Check your environment variables.")

        openai.api_base = "http://aiproxy.sanand.workers.dev/openai/v1"
        openai.api_key = token

        # 🔹 Get embeddings for all comments using GPT-4o-mini
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Generate embeddings for these comments: {json.dumps(comments)}"}]
        )

        # 🔹 Extract embeddings from the response
        try:
            embeddings = np.array(json.loads(response["choices"][0]["message"]["content"])["embeddings"])
        except (json.JSONDecodeError, KeyError, ValueError):
            raise Exception(f"⚠️ Failed to parse embeddings: {response}")

        # 🔹 Compute cosine similarity between all pairs
        num_comments = len(comments)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Set self-similarity (diagonal elements) to -inf to ignore them
        np.fill_diagonal(similarity_matrix, -np.inf)

        # 🔹 Get the most similar pair of comments
        i, j = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
        best_pair = (comments[i], comments[j])

        # 🔹 Write the most similar comments to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(best_pair)) + "\n")  # Sort for consistent ordering

        return {"status": "success", "message": f"Most similar comments saved to {output_file}", "comments": best_pair}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_markdown_titles():
    try:
        # 🔹 Define the local `docs/` directory
        docs_dir = os.path.join(os.getcwd(), "data", "docs")
        output_file = os.path.join(os.getcwd(), "data", "docs", "index.json")

        # 🔹 Ensure the `docs/` directory exists
        if not os.path.exists(docs_dir):
            raise Exception(f"Docs directory not found: {docs_dir}")

        # 🔹 Dictionary to store extracted titles
        index = {}

        # 🔹 Process all Markdown (`.md`) files
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)

                    # 🔹 Extract the first H1 title
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("# "):  # H1 heading found
                                title = line[2:].strip()
                                relative_path = os.path.relpath(file_path, docs_dir).replace("\\", "/")  # Store relative path
                                index[relative_path] = title
                                break  # Stop after the first H1

        # 🔹 Write the index JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)

        return {"status": "success", "message": f"Extracted H1 titles from {len(index)} markdown files.", "output_file": output_file}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run")
def run(task: str = Query(..., description="Task to execute")):
    """Executes the given task."""
    return run_task(task)

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(...)):
    """
    GET endpoint to read and return the content of a file.
    Ensures only files under /data (as specified in the task) are accessed.
    """
    # Security check: Path must start with /data
    if not path.startswith("/data"):
        raise HTTPException(status_code=400, detail="Invalid file path: Must start with /data")
    
    # Translate the given path into a local path.
    # Assuming your repository has a 'data' folder in its root,
    # we remove the leading '/data' and join with the repository's data directory.
    base_dir = os.path.join(os.getcwd(), "data")  # local data folder
    relative_path = os.path.relpath(path, "/data")  # e.g. "sample.txt"
    file_path = os.path.join(base_dir, relative_path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")