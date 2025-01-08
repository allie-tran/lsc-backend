"""
All utilities related to question answering
"""

import re
from collections import defaultdict

from configs import QUERY_PARSER
from llm import gpt_llm_model, llm_model
from llm.prompt.parse import PARSE_QUERY, REWRITE_QUESTION
from rich import print as rprint

from query_parse.types.lifelog import ParsedQuery, SingleQuery

from .constants import AUXILIARY_VERBS, QUESTION_WORDS, STOP_WORDS


def detect_question(query: str) -> bool:
    """
    Check if the query is a question or not
    This is useful in case the user forgets to toogle question mode
    """
    if "?" in query:
        return True

    # detect mutli-sentence questions
    sentences = re.split(r"[.!?,]", query)
    for sentence in sentences:
        words = sentence.lower().strip().split()
        if words:
            # check if the first word is a question word
            if words[0] in QUESTION_WORDS:
                return True

            if len(words) > 1:
                if words[0] in AUXILIARY_VERBS and words[1] in QUESTION_WORDS:
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
    if isinstance(search_text, dict) and "text" in search_text:
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


async def parse_query(text: str, is_question: bool) -> ParsedQuery:
    """
    Get the relevant fields from the query
    """
    template = {
        "main": defaultdict(lambda: text),
        "after": defaultdict(str),
        "before": defaultdict(str),
        "must_not": defaultdict(str),
    }

    if QUERY_PARSER or is_question:
        # in some cases, it's inefficient to parse the query
        if detect_simple_query(text):
            main = SingleQuery(visual=text, location=text, time=text, date=text)
            return ParsedQuery(main=main)

        prompt = PARSE_QUERY.format(query=text)
        feat = await gpt_llm_model.generate_from_text(prompt)

        if isinstance(feat, dict):
            for key, value in feat.items():
                if key in template:
                    query = template[key]

                    for k, v in value.items():
                        query[k] = v

                    # add location into main
                    if "location" in query and "visual" in query:
                        if query["location"] != query["visual"]:
                            query["visual"] = query["visual"] + " " + query["location"]

        else:
            print("Failed to parse query")
            print(feat)

    parsed_query = ParsedQuery.model_validate(template)
    rprint(parsed_query)
    return parsed_query
