import asyncio
import os
from collections.abc import AsyncGenerator
from typing import List

from async_timeout import timeout
from nltk.probability import random
from configs import IMAGE_DIRECTORY, TIMEOUT
from llm import internlm_model
from results.models import Event, EventResults

from question_answering.text import format_answer


async def answer_visual_only(
    question: str, textual_descriptions: List[str], events: EventResults, k: int = 10
) -> AsyncGenerator[str, None]:
    """
    Answer the question using the visual information only
    K: number of events to consider
    """
    if not internlm_model.loaded:
        internlm_model.load_model()

    # Get the list of images
    try:
        tasks = []
        for i, (event, description) in enumerate(zip(events.events[:k], textual_descriptions[:k])):
            tasks.append(answer_visual_one_event(i, question, description, event))

        async with timeout(TIMEOUT):
            for future in asyncio.as_completed(tasks):
                answer = await future
                if answer:
                    for ans in answer:
                        yield ans
    except asyncio.TimeoutError:
        yield "Timeout"


black_list = [
    "I'm sorry",
    "I'm not sure",
    "I don't know",
    "I can't answer",
    "AI language model",
    "I cannot",
    "I can't",
    "I am sorry",
    "I am not sure",
    "I am not able",
    "The image does not",
    "The image is not",
    "The image is blurry",
    "The image is unclear",
    "The image doesn't",
    "There are no",
    "The image you provided",
    "The image is too",
    "blurry",
    "low-quality",
    "blurred"
]


def is_black_listed(answer: str) -> bool:
    """
    Check if the answer is in the black list
    """
    if not answer:
        return True
    for black in black_list:
        if black in answer:
            return True
    return False


async def answer_visual_one_event(n: int,
    question: str, textual_description: str, event: Event,
) -> List[str]:
    """
    Process the question for a single event
    """
    image_paths = [os.path.join(IMAGE_DIRECTORY, img) for img in event.images]
    if len(image_paths) == 0:
        return []
    valid_answers = {}
    loops = []
    if len(image_paths) < 3:
        loops = [image_paths]
    else:
        for _ in range(3):
            samples = random.sample(image_paths, k=2)
            loops.append(samples)
    for samples in loops:
        answer_dict = await internlm_model.answer_question(
            question, samples, extra_info=textual_description
        )
        for answer in answer_dict:
            if not is_black_listed(answer_dict[answer]):
                valid_answers[answer] = f"[Event {n}] {answer_dict[answer]}"
    if not valid_answers:
        return []
    return format_answer(valid_answers)
