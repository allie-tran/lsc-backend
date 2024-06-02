import asyncio
import base64
import os
from collections.abc import AsyncGenerator
from typing import List

from async_timeout import timeout
from configs import IMAGE_DIRECTORY, TIMEOUT
from llm import gpt_llm_model, internlm_model
from llm.models import MixedContent
from llm.prompts import MIXED_PROMPTS
from nltk.probability import random
from results.models import AnswerResult, Event, GenericEventResults


async def answer_visual_only(
    question: str,
    textual_descriptions: List[str],
    events: GenericEventResults,
    k: int = 10,
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
        for i, (event, description) in enumerate(
            zip(events.events[:k], textual_descriptions[:k])
        ):
            e = event if isinstance(event, Event) else event.main
            tasks.append(answer_visual_one_event(i, question, description, e))

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
    "blurred",
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


async def answer_visual_one_event(
    n: int,
    question: str,
    textual_description: str,
    event: Event,
) -> AsyncGenerator[List[AnswerResult], None]:
    """
    Process the question for a single event
    """
    image_paths = [os.path.join(IMAGE_DIRECTORY, img.src) for img in event.images]
    if len(image_paths) == 0:
        yield []
        return
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
                yield [
                    AnswerResult(
                        text="MMLM",
                        explanation=[answer],
                        evidence=[n],
                    )
                ]
    yield []

def to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


async def answer_visual_with_text(
    question: str,
    textual_descriptions: List[str],
    results: GenericEventResults,
    k: int = 10,
) -> AsyncGenerator[List[AnswerResult], None]:
    """
    Given a natural language question and a list of scenes, returns the top k answers
    Note that the EventResults have already filtered the relevant fields
    """
    ## First get the textual description of the events
    content: List[MixedContent] = [
        MixedContent(
            type="text",
            content=MIXED_PROMPTS[0].format(question=question, num_events=k),
        )
    ]
    for i, text in enumerate(textual_descriptions):
        image_paths = results.events[i].images
        images = [os.path.join(IMAGE_DIRECTORY, img.src) for img in image_paths][:3]
        bs64_images = [to_base64(image) for image in images if os.path.exists(image)]
        description = f"Event {i+1}. {text}"

        content.append(MixedContent(type="text", content=description))
        content.extend(
            [MixedContent(type="image_url", content=image) for image in bs64_images]
        )
    content.append(
        MixedContent(type="text", content=MIXED_PROMPTS[1].format(question=question))
    )

    async for llm_response in gpt_llm_model.generate_from_mixed_media(content):
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
            print("GPT", llm_response)
            pass
