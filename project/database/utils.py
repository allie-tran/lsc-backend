from typing import List, Optional, Tuple

from configs import ESSENTIAL_FIELDS
from llm.gpt import llm_model
from llm.prompts import QA_PROMPT, RELEVANT_FIELDS_PROMPT
from pymongo import MongoClient
from query_parse.utils import extend_no_duplicates
from question_answering.text import get_specific_description
from results.models import Event
from rich import print

client = MongoClient("localhost", 27017)
db = client["LSC24"]

scene_collection = db["scenes"]
image_collection = db["images"]
location_collection = db["locations"]
user_collection = db["users"]


def convert_to_events(
    event_list: List[str],
    relevant_fields: Optional[List[str]] = None,
) -> List[Event]:
    """
    Convert a list of event ids to a list of Event objects
    """
    documents = []
    if relevant_fields:
        projection = extend_no_duplicates(relevant_fields, ESSENTIAL_FIELDS)
        try:
            documents = scene_collection.find(
                {"scene": {"$in": event_list}}, projection=projection
            )
        except Exception as e:
            print("[red]Error in convert_to_events[/red]", e)

    if not documents:
        documents = scene_collection.find({"scene": {"$in": event_list}})
    return [Event(**doc) for doc in documents]


async def get_relevant_fields(query: str) -> Tuple[List[str], str]:
    """
    Get the relevant fields from the query
    """
    prompt = RELEVANT_FIELDS_PROMPT.format(query=query)
    data = await llm_model.generate_from_text(prompt)
    if isinstance(data, list):
        print(f"[green]Found list![/green] {data}")
    else:
        print("[red]No list found! Including images and scene only.[/red]")
        data = []
    return data, "llm"


async def text_qa_answer(question: str, event: Event):
    """
    Extract textual information from the scene
    then answer the question using the QA model
    """
    if event.images:
        ocr = image_collection.find({"image": {"$in": event.images}}, {"ocr": 1})
        ocr = set([text for img in ocr for text in img["ocr"]])
        if "" in ocr:
            ocr.remove("")
        event.ocr = list(ocr)

    answers = {}
    textual_description = ""
    try:
        textual_description = get_specific_description(event)
        QA_input = {"question": question, "context": textual_description}
        prompt = QA_PROMPT.format(question=question, events=textual_description)

        print(prompt)

        # Get answers from textual description
        answers = await llm_model.generate_from_text(prompt)

    except Exception as e:
        print("Error in TextQA", e)
        raise (e)

    return textual_description, answers
