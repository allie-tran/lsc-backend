from typing import List

from results.database import scene_collection
from results.models import Event


def convert_to_events(event_list: List[str]) -> List[Event]:
    """
    Convert a list of event ids to a list of Event objects
    """
    documents = scene_collection.find({"scene": {"$in": event_list}})
    return [Event(**doc) for doc in documents]
