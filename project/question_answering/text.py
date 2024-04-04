# Get general textual description for a scene

from typing import List, Optional

from configs import DURATION_FIELDS, LOCATION_FIELDS, TIME_FIELDS
from llm.prompts import QA_PROMPT
from llm.gpt import llm_model
from query_parse.time import calculate_duration
from results.models import Event, EventResults


def get_general_textual_description(event: Event) -> str:
    # Default values
    time = ""
    duration = ""
    start_time = ""
    end_time = ""
    date = ""
    weekday = ""
    location = ""
    location_info = ""
    region = ""
    ocr = ""

    # Start calculating
    duration = calculate_duration(event.start_time, event.end_time)

    # converting datetime to string
    start_time = event.start_time.strftime("%I:%M%p")
    end_time = event.end_time.strftime("%I:%M%p")
    time = f"from {start_time} to {end_time} "

    # converting datetime to string like Jan 1, 2020
    date = event.start_time.strftime("%A, %B %d, %Y")

    # Location
    if event.location:
        location = f"in {event.location}"
    if event.location_info:
        location_info = f", which is a {event.location_info}"

    # Region
    if event.region:
        regions = [reg for reg in event.region if reg != event.country]
        region = f"in {', '.join(regions)}"
    if event.country:
        region += f" in {event.country}"

    # OCR
    if event.ocr:
        ocr = f"Some texts that can be seen from the images are: {' '.join(event.ocr)}."

    textual_description = (
        f"The event happened {time}{duration} on {date} "
        + f"{location}{location_info} in {region} in {event.country}. {ocr}"
    )
    return textual_description


# Get textual description for a scene
def get_specific_description(event: Event, fields: Optional[List[str]] = None) -> str:
    if fields is None:
        return get_general_textual_description(event)

    # Default values
    time = ""
    duration = ""
    location = ""
    visual = ""

    # Start calculating
    for field in TIME_FIELDS:
        if field in fields:
            time += f"at {field} {getattr(event, field)} "

    # Duration
    for field in DURATION_FIELDS:
        if field in fields:
            duration += f"{getattr(event, field)} {field}"

    if duration:
        duration = "which lasted for " + duration + " "

    # Location
    for field in LOCATION_FIELDS:
        if field in fields:
            location += f"in {getattr(event, field)} "

    # Visual
    if event.ocr:
        visual = (
            f"Some texts that can be seen from the images are: {' '.join(event.ocr)}."
        )

    textual_description = f"The event happened {time}{duration} {location}. {visual}"
    return textual_description


async def answer_text_only(question: str, events: EventResults, k: int = 10) -> str:
    """
    Given a natural language question and a list of scenes, returns the top k answers
    Note that the EventResults have already filtered the relevant fields
    """
    ## First get the textual description of the events
    k = min(k, len(events.events))
    textual_descriptions = []
    if events.relevant_fields:
        for i, event in enumerate(events.events[:k]):
            textual_descriptions.append(
                get_specific_description(event, events.relevant_fields)
            )
            # text = event.model_dump(include=set(events.relevant_fields))
            # textual_descriptions.append(text)
        print("[green]Textual description sample[/green]", textual_descriptions[0])

    formated_textual_descriptions = ""
    for i, text in enumerate(textual_descriptions):
        formated_textual_descriptions += f"{i+1}. {text}\n"

    prompt = QA_PROMPT.format(
        question=question,
        num_events=len(events.events),
        events=formated_textual_descriptions,
    )

    answers = await llm_model.generate_from_text(prompt)
    if isinstance(answers, str):
        return answers

    answers = "\n".join(f"{k}({v})" for k, v in answers.items())
    return answers
