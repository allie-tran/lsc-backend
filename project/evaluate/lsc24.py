import json
import logging

from nltk import os
import pandas as pd
from fastapi.testclient import TestClient
from main import app

filename = "evaluate/Lifelog QA Test collection - Questions.csv"
questions = pd.read_csv(filename)

# questions.head()
# questions.columns
# reverse questions
questions = questions.iloc[::-1]

client = TestClient(app)

# |%%--%%| <qZmKMAlnM8|daKW0SXRKX>

logging.basicConfig(level=logging.INFO)

# log to file
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler("evaluate/lsc24.log")
fh.setLevel(logging.INFO)
fh.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(fh)
for i, row in questions.iterrows():
    row = row.to_dict()
    question_id = row["ID"]
    if os.path.exists(f"evaluate/{question_id}/done.txt"):
        continue
    os.makedirs(f"evaluate/{question_id}", exist_ok=True)

    # Save the question to a file
    question: str = row["Question"]
    answer = row["Answer"]
    other_answers = row["Other answers"]
    if pd.isnull(other_answers):
        other_answers = []
    else:
        other_answers = other_answers.split(",")
        other_answers = [x.strip() for x in other_answers]

    logger.info("-" * 50)
    logger.info(question)
    logger.info(answer)
    logger.info(", ".join(other_answers))

    with open(f"evaluate/{question_id}/answers.txt", "a") as f:
        f.write("-" * 50 + "\n")
        f.write(question + "\n")
        f.write(answer + "\n")
        f.write(", ".join(other_answers) + "\n")

    # Send the question to the server
    response = client.post("/search", json={"main": question})
    token = response.json()["searchToken"]

    # Don't go to the next question until the results are ready
    with client.stream("GET", f"/get-stream-results/test/{token}") as response:
        assert response.status_code == 200
        for text in response.iter_lines():
            # Remove data: from the beginning of the line
            text = text.replace("data: ", "").strip()
            if not text:
                continue
            if text == "END":
                logger.info("End of stream")
                break
            else:
                data = json.loads(text)
                if data["type"] == "raw":
                    logger.info("Raw data")
                    logger.info(json.dumps(data["results"], indent=2))

                if data["type"] == "modified":
                    logger.info("Processed data")
                    logger.info(json.dumps(data["results"], indent=2))

                if data["type"] == "answers":
                    logger.info("Answers")
                    logger.info(json.dumps(data["answers"]))

                    with open(f"evaluate/{question_id}/clean.log", "a") as f:
                        f.write("Generated answers\n")
                        f.write(json.dumps(data) + "\n")

    to_continue = input("Continue? (y/n): ")
    if to_continue.lower() != "y":
        break
