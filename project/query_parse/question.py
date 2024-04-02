"""
All utilities related to question answering
"""

from .constants import AUXILIARY_VERBS, QUESTION_TYPES, QUESTION_WORDS


def detect_question(query: str) -> bool:
    """
    Check if the query is a question or not
    This is useful in case the user forgets to toogle question mode
    """

    if query.endswith("?"):
        return True
    words = query.split()
    if words[0].lower() in QUESTION_WORDS:
        return True
    return False


def question_to_retrieval(query: str) -> str:
    """
    Convert a question to a retrieval query
    """
    # TODO: Implement this function
    # For now, we just return the query as it is

    # Remove the question word
    words = query.split()
    if words[0].lower() in QUESTION_WORDS:
        words = words[1:]

    # remove the second word if it is a auxiliary verb
    if words[0].lower() in AUXILIARY_VERBS:
        words = words[1:]

    query = " ".join(words)
    print("Converted query:", query)
    return query


def question_classification(query: str) -> str:
    """
    Classify the question type
    """
    # TODO!
    for question_type in QUESTION_TYPES:
        if question_type in query:
            return question_type
    return "other"
