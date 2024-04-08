"""
All utilities related to question answering
"""

from llm.prompts import REWRITE_QUESTION
from .constants import AUXILIARY_VERBS, QUESTION_TYPES, QUESTION_WORDS

from llm import llm_model

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
            parts = query.lower().split(sep)
            if parts[0].strip() in QUESTION_WORDS:
                return True

    return False


async def question_to_retrieval(query: str) -> str:
    """
    Convert a question to a retrieval query
    """
    # TODO: Implement this function
    # For now, we just return the query as it is

    prompt = REWRITE_QUESTION.format(question=query)
    search_text = await llm_model.generate_from_text(prompt)
    if isinstance(search_text, dict):
        search_text = search_text["text"]
    return search_text


def question_classification(query: str) -> str:
    """
    Classify the question type
    """
    # TODO!
    for question_type in QUESTION_TYPES:
        if question_type in query:
            return question_type
    return "other"
