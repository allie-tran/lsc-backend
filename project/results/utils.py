from collections import defaultdict
from functools import cmp_to_key
from typing import Iterable, List, Union

from configs import DERIVABLE_FIELDS, ISEQUAL
from query_parse.visual import score_images
from retrieval.common_nn import encode_query
from rich import print

from results.models import DerivedEvent, Event, EventResults


def deriving_fields(
    events: Union[List[Event], List[DerivedEvent]], fields: List[str]
) -> List[DerivedEvent]:
    """
    Derive the fields
    """
    derived_events = []
    for event in events:
        derived_event = DerivedEvent(**event.dict())
        for field in fields:
            setattr(derived_event, field, DERIVABLE_FIELDS[field](event))
        derived_events.append(derived_event)
    return derived_events


def custom_compare_function(event1: Event, event2: Event, fields: Iterable[str]) -> int:
    """
    Custom compare function
    """
    # Assert if the two events are the same group first
    # if not then just compare the scene ids
    equal = True
    for field in fields:
        cmp_field = field
        if field not in ISEQUAL:
            cmp_field = "*"
        if not ISEQUAL[cmp_field](getattr(event1, field), getattr(event2, field)):
            equal = False
            break
    if equal:
        return 0

    # If the two events are not the same group then compare the sceneid
    return 1 if event1.scene > event2.scene else -1

def merge_events(results: EventResults) -> EventResults:
    """
    Merge the events
    """
    groupby = set()
    if results.relevant_fields:
        for field in results.relevant_fields:
            if field.startswith("groupby_"):
                groupby.add(field.split("_")[1])

    # Check if any of the groupby should be calculated
    available_fields = set(type(results.events[0]).model_fields)
    derive_fields = []
    to_remove = set()
    for criteria in groupby:
        if criteria not in available_fields:
            # Two cases here:
            # 1. the field is derivable from the schema
            if criteria in DERIVABLE_FIELDS:
                print("[blue]Deriving field[/blue]", criteria)
                derive_fields.append(criteria)
            # 2. the field is not derivable from the schema
            else:
                print("[red]Cannot derive field[/red]", criteria)
                to_remove.add(criteria)

    # Remove the fields that cannot be derived
    groupby = groupby.difference(to_remove)

    # Derive the fields
    if derive_fields:
        results.events = deriving_fields(results.events, derive_fields)

    # If groupby is empty then group by "group"
    if not groupby:
        groupby = ["group"]

    cmp = lambda x, y: custom_compare_function(x, y, groupby)

    # Group the events using the ISEQUAL criteria from configs
    temp = results.events
    sorted(temp, key=cmp_to_key(cmp))

    # Get unique keys
    unique_groups = 0
    event_to_group = {temp[0].scene: 0}
    for event1, event2 in zip(temp, temp[1:]):
        if cmp(event1, event2) != 0:
            unique_groups += 1
        event_to_group[event2.scene] = unique_groups

    print(f"[blue]Grouping {len(results.events)} events by {groupby}[/blue]")
    scores = results.scores
    grouped_events = defaultdict(list)
    grouped_scores = defaultdict(list)

    # Group the events
    for event, score in zip(results.events, results.scores):
        grouped_events[event_to_group[event.scene]].append(event)
        grouped_scores[event_to_group[event.scene]].append(score)

    new_results = []
    new_scores = []
    for group in grouped_events:
        # Merge the events
        events = grouped_events[group]
        scores = grouped_scores[group]

        if len(events) == 1:
            new_results.append(events[0])
            new_scores.append(scores[0])
            continue

        new_event = events[0]
        new_event.merge_with_many(scores[0], events[1:], scores=scores[1:])
        new_results.append(new_event)
        new_scores.append(grouped_scores[group][0])

    return EventResults(
        events=new_results,
        scores=new_scores,
        relevant_fields=results.relevant_fields + derive_fields,
        min_score=results.min_score,
        max_score=results.max_score,
    )


def limit_images_per_event(
    results: EventResults, text_query: str, max_images: int
) -> EventResults:
    """
    Limit the number of images per event
    This is achieved by selecting the images with the highest score
    and highest relevance to the query
    """
    for event in results.events:
        images = event.images
        scores = event.image_scores

        if len(images) <= max_images:
            continue

        encoded_query = encode_query(text_query)
        visual_scores = score_images(images, encoded_query)

        # Sort the images by the visual score and the original score
        ensembled_scores = [
            (visual_score + 1) * (score + 1)
            for visual_score, score in zip(visual_scores, scores)
        ]

        sorted_images = [
            x for _, x in sorted(zip(ensembled_scores, images), reverse=True)
        ]
        chosen_images = sorted_images[:max_images]
        indices = sorted([images.index(image) for image in chosen_images])

        new_images = [images[i] for i in indices]
        new_scores = [scores[i] for i in indices]

        # Update the event
        event.images = new_images
        event.image_scores = new_scores

    return results


def basic_label(event: Event) -> str:
    """
    Create a basic label for the event
    """
    start_time = event.start_time.strftime("%H:%M")
    end_time = event.end_time.strftime("%H:%M")
    location = event.location
    date = event.start_time.strftime("%d, %b %Y")
    return f"<b>{location}</b>\n{date}, {start_time} - {end_time}"


def create_event_label(results: EventResults) -> EventResults:
    """
    Create a label for the event
    """
    if_empty_fields = not results.relevant_fields
    no_groupby = not any([field.startswith("groupby_") for field in results.relevant_fields])

    if if_empty_fields or no_groupby:
        # Get the basic fields: location, time, date
        for event in results.events:
            event.name = basic_label(event)
    else:
        important_fields = set()
        normal_fields = set()
        for field in results.relevant_fields:
            if field.startswith("groupby_"):
                field = field.split("_")[1:]
                continue
            if field.startswith("sortby_"):
                continue
            if field in ["group", "gps", "timestamp", "scene", "images", "ocr"]:
                continue
            if field in DERIVABLE_FIELDS:
                important_fields.add(field)
                continue
            if field in Event.model_fields:
                normal_fields.add(field)

        all_fields = list(important_fields) + list(normal_fields)
        if len(all_fields) > 3:
            all_fields = all_fields[:3]

        for event in results.events:
            label = []
            for field in important_fields:
                if field in ["start_time", "end_time"]:
                    value = getattr(event, field).strftime("%H:%M")
                else:
                    value = getattr(event, field)
                if len(value) > 50:
                    value = value[:50] + "..."
                label.append(f"{field.capitalize()}: {getattr(event, field).capitalize()}")
            event.name = "\n".join(label)
    return results
