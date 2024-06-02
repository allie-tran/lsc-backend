"""
All utilities related to question answering
"""

from collections import defaultdict
from typing import Dict

from configs import QUERY_PARSER
from llm import llm_model
from llm.prompts import PARSE_QUERY, REWRITE_QUESTION

from .constants import QUESTION_TYPES, QUESTION_WORDS, STOP_WORDS


def detect_question(query: str) -> bool:
    """
    Check if the query is a question or not
    This is useful in case the user forgets to toogle question mode
    """
    if "?" in query:
        return True

    # detect mutli-sentence questions
    seperators = [".", "!", ";", "\n"]
    for sep in seperators:
        if sep in query:
            sentences = query.split(sep)
            for sentence in sentences:
                first_word = sentence.split()[0]
                if first_word in QUESTION_WORDS:
                    return True
                if len(sentence.split()) > 1:
                    second_word = sentence.split()[1]
                    if second_word in QUESTION_WORDS:
                        return True
    return False


async def question_to_retrieval(text: str, is_question: bool) -> str:
    """
    Convert a question to a retrieval query
    """
    if not is_question:
        return text

    prompt = REWRITE_QUESTION.format(question=text)
    search_text = await llm_model.generate_from_text(prompt)
    if isinstance(search_text, dict):
        search_text = search_text["text"]
    if search_text and isinstance(search_text, str):
        return search_text
    return text


def detect_simple_query(query: str) -> bool:
    """
    Check if the query is a simple query or not
    if it's too short, or full of stop words, it's a simple query
    """
    words = query.split()
    words = [word for word in words if word not in STOP_WORDS]
    return len(words) < 3


async def parse_query(text: str) -> Dict[str, str]:
    """
    Get the relevant fields from the query
    """
    template = defaultdict(lambda: text)
    if QUERY_PARSER:
        # in some cases, it's inefficient to parse the query
        if detect_simple_query(text):
            return template

        prompt = PARSE_QUERY.format(query=text)
        feat = await llm_model.generate_from_text(prompt)

        if isinstance(feat, dict):
            print(feat)
            for key, value in feat.items():
                template[key] = value
            # add location into visual
            if "location" in template and "visual" in template:
                if template["location"] != template["visual"]:
                    template["visual"] = template["visual"] + " " + template["location"]
        else:
            print("Failed to parse query")
            print(feat)

    return template


def question_classification(query: str) -> str:
    """
    Classify the question type
    """
    # TODO!
    for question_type in QUESTION_TYPES:
        if question_type in query:
            return question_type
    return "other"
