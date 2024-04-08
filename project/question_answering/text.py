# Get general textual description for a scene

from collections.abc import AsyncGenerator
from typing import Dict, List, Optional

from configs import DURATION_FIELDS, LOCATION_FIELDS, TIME_FIELDS
from llm import llm_model
from llm.prompts import QA_PROMPT
from query_parse.time import calculate_duration
from results.models import Event


def get_general_textual_description(event: Event) -> str:
    # Default values
    time = ""
    duration = ""
    start_time = ""
    end_time = ""
    date = ""
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
            if field == "start_time":
                time += f" from {getattr(event, field).strftime('%H:%M')}"
            elif field == "end_time":
                time += f" to {getattr(event, field).strftime('%H:%M')}"
            else:
                time += f" at {field} {getattr(event, field)}"

    # Duration
    for field in DURATION_FIELDS:
        if field in fields:
            duration += f"{getattr(event, field)} {field}"

    if duration:
        duration = " which lasted for " + duration + " "

    # Location
    for field in LOCATION_FIELDS:
        if field in fields:
            value = getattr(event, field)
            if value:
                location += f" in {getattr(event, field)} "

    # Visual
    if event.ocr:
        visual = (
            f"Some texts that can be seen from the images are: {' '.join(event.ocr)}."
        )

    textual_description = f"This event happened{time}{duration}{location}. {visual}"
    return textual_description


async def answer_text_only(
    question: str, textual_descriptions: List[str], num_events: int
) -> AsyncGenerator[List[str], None]:
    """
    Given a natural language question and a list of scenes, returns the top k answers
    Note that the EventResults have already filtered the relevant fields
    """
    ## First get the textual description of the events

    formated_textual_descriptions = ""
    for i, text in enumerate(textual_descriptions):
        formated_textual_descriptions += f"Event {i+1}. {text}\n"

    prompt = QA_PROMPT.format(
        question=question,
        num_events=num_events,
        events=formated_textual_descriptions,
    )

    async for llm_response in llm_model.stream_from_text(prompt):
        if not isinstance(llm_response, str) and "answers" in llm_response:
            if llm_response["answers"]:
                yield format_answer(llm_response["answers"])


def format_answer(answers: List[Dict]) -> List[str]:
    """
    Format the answers into a string
    """
    formatted_answers = []

    for answer_dict in answers:
        try:
            answer = answer_dict["answer"]
            explanation = answer_dict["explanation"]
            evidence = answer_dict["evidence"]

            formatted = f"<strong class='answer'>{answer}</strong>\n{explanation}\n"

            evidence_str = ""
            for ev in evidence:
                evidence_str += f"<span class='evidence' data={ev}>Event {ev}</span>\n"

            formatted += evidence_str
            formatted_answers.append(formatted)
        except Exception:
            pass

    return formatted_answers

