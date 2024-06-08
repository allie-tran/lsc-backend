import asyncio
import base64
import os
import random
from collections.abc import AsyncGenerator
from typing import List

from configs import IMAGE_DIRECTORY
from llm import gpt_llm_model
from llm.models import MixedContent
from llm.prompts import MIXED_PROMPTS
from results.models import AnswerResult, Event, GenericEventResults, Image
from retrieval.async_utils import async_generator_timer
from rich import print as rprint


async def answer_visual_only(
    question: str,
    textual_descriptions: List[str],
    results: GenericEventResults,
    k: int = 10,
) -> AsyncGenerator[List[AnswerResult], None]:
    """
    Answer the question using the visual information only
    K: number of events to consider
    """
    # if internlm_model.loaded:
    #     internlm_model.load_model()

    # Get the list of images
    try:
        for i, (event, description) in enumerate(
            zip(results.events[:k], textual_descriptions[:k])
        ):
            e = event if isinstance(event, Event) else event.main
            async for answers in answer_visual_one_event(i, question, description, e):
                yield answers

    except asyncio.TimeoutError:
        yield []


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
    image_paths = event.images
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
        async for answer_list in answer_visual_with_text(
            question, samples, textual_description
        ):
            for answer_dict in answer_list:
                try:
                    answer: str = answer_dict["answer"]
                    explanation: str = answer_dict["explanation"]
                    if not is_black_listed(answer):
                        yield [
                            AnswerResult(
                                text=answer,
                                explanation=[explanation],
                                evidence=[n + 1],
                            )
                        ]
                except Exception as e:
                    rprint(e)
                    rprint("GPT", answer_dict)
    yield []


def to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


@async_generator_timer("answer_visual_with_text")
async def answer_visual_with_text(
    question: str, image_paths: List[Image], textual_description: str
) -> AsyncGenerator[list[dict], None]:
    """
    Given a natural language question and a list of scenes, returns the top k answers
    Note that the EventResults have already filtered the relevant fields
    """
    images = [os.path.join(IMAGE_DIRECTORY, img.src) for img in image_paths][:3]
    bs64_images = [to_base64(image) for image in images if os.path.exists(image)]
    content = [MixedContent(type="image_url", content=image) for image in bs64_images]
    content.append(
        MixedContent(
            type="text",
            content=MIXED_PROMPTS.format(
                question=question, extra_info=textual_description
            ),
        )
    )
    task = gpt_llm_model.generate_from_mixed_media(content)
    async for llm_response in task:
        try:
            if llm_response and "answers" in llm_response:
                yield llm_response["answers"]
        except Exception as e:
            rprint(e)
            rprint("GPT", llm_response)
