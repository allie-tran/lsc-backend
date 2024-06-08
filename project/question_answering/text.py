# Get general textual description for a scene

from collections.abc import AsyncGenerator
from typing import List, Optional

from configs import DURATION_FIELDS, LOCATION_FIELDS, TIME_FIELDS
from llm import llm_model
from llm.prompts import QA_PROMPT
from query_parse.time import calculate_duration
from results.models import AnswerListResult, AnswerResult, Event


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
) -> AsyncGenerator[List[AnswerResult], None]:
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
        if llm_response:
            try:
                answers = [
                    AnswerResult(
                        text=answer["answer"],
                        explanation=[answer["explanation"]],
                        evidence=[int(ev) for ev in answer["evidence"]],
                    )
                    for answer in llm_response["answers"]
                ]
                yield answers
            except Exception:
                print("GROQ", llm_response)
                pass


def format_answer(answers: AnswerListResult) -> List[str]:
    """
    Format the answers into a string
    """
    formatted_answers = []
    for answer, data in answers.answers.items():
        try:
            explanation = "\n".join(data.explanation)
            evidence = data.evidence

            formatted = f"<strong class='answer'>{answer}</strong>\n{explanation}\n"

            evidence_str = []
            for ev in evidence:
                evidence_str.append(
                    f"<span class='evidence' data={ev}>Event {ev}</span>"
                )
            evidence_str = ", ".join(evidence_str)

            formatted += evidence_str
            formatted_answers.append(formatted)
        except Exception as e:
            raise (e)

    return formatted_answers
