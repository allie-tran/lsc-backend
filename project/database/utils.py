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
    index = {scene: i for i, scene in enumerate(event_list)}
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

    # Sort the documents based on the order of the event_list
    documents = sorted(documents, key=lambda doc: index[doc["scene"]])
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
