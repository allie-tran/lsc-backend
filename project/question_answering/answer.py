import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from configs import (
    DURATION_FIELDS,
    LOCATION_FIELDS,
    QA_DIM,
    QA_FEATURES,
    QA_IDS,
    TIME_FIELDS,
)
from llm.gpt import llm_model
from llm.prompts import QA_PROMPT
from query_parse.constants import BASIC_DICT
from query_parse.time import calculate_duration
from results.models import Event, EventResults
from rich import print
from transformers.utils import logging as transformers_logging

from .FrozenBiLM.util.misc import get_mask
from .qa_models import args, device, qa_model

transformers_logging.set_verbosity_error()

# Load QA features
qa_photo_features = []
qa_photo_ids = []
for feat, ids in zip(QA_FEATURES, QA_IDS):
    assert os.path.exists(feat), f"File {feat} not found"
    assert os.path.exists(ids), f"File {ids} not found"
    qa_photo_features.append(np.load(feat))
    qa_photo_ids.extend(pd.read_csv(ids)["photo_id"].to_list())  # type: ignore
qa_photo_features = np.concatenate(qa_photo_features)
image_to_id = {image: i for i, image in enumerate(qa_photo_ids)}


def answer(
    images: List[str],
    encoded_question: Dict[str, torch.Tensor],
) -> Dict[str, str]:
    """
    Given a list of image filenames and an encoded question, generates the top 5 answers
    to the question based on the provided image embeddings.
    """
    assert qa_model.loaded, "Model not loaded"
    assert len(images) > 0, "No images provided"

    # Get image embeddings for the provided images
    image_embeddings = qa_photo_features[
        np.array([image_to_id[image] for image in images if image])
    ]

    # Convert image embeddings to PyTorch tensor and move to appropriate device
    video = torch.tensor(image_embeddings).to(device).float()

    # Subsample or pad the tensor if its length exceeds max_feats
    if len(video) >= args.max_feats:
        sampled = []
        for j in range(args.max_feats):
            sampled.append(video[(j * len(video)) // args.max_feats])
        video = torch.stack(sampled)
        video_len = args.max_feats
    else:
        video_len = len(video)
        video = torch.cat(
            [video, torch.zeros(args.max_feats - video_len, QA_DIM).to(device)], 0
        )

    # Add an additional dimension to the tensor and move to appropriate device
    video = video.unsqueeze(0).to(device)

    # Create a mask for the tensor
    video_mask = get_mask(
        torch.tensor(video_len, dtype=torch.long).unsqueeze(0), video.size(1)
    ).to(device)

    # Move encoded question to appropriate device
    input_ids = encoded_question["input_ids"].to(device)
    attention_mask = encoded_question["attention_mask"].to(device)

    # Remove separator token and replace with padding token if not using suffix
    if not args.suffix:
        attention_mask[input_ids == qa_model.tokenizer.sep_token_id] = 0
        input_ids[input_ids == qa_model.tokenizer.sep_token_id] = (
            qa_model.tokenizer.pad_token_id
        )

    # Use the provided BiLM to generate a prediction for the mask token
    output = qa_model.visual_qa(
        video=video,
        video_mask=video_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Extract the logits for the mask token
    logits = output["logits"]
    delay = args.max_feats if args.use_video else 0
    logits = logits[:, delay : encoded_question["input_ids"].size(1) + delay][
        encoded_question["input_ids"] == qa_model.tokenizer.mask_token_id
    ]  # get the prediction on the mask token
    logits = logits.softmax(-1)
    try:
        topk = torch.topk(logits, 5, -1)
        topk_txt = [
            [qa_model.id2a[int(x.item())] for x in y] for y in topk.indices.cpu()
        ]
        topk_scores = [[f"{x:.2f}".format() for x in y] for y in topk.values.cpu()]
        topk_all = [{x: y for x, y in zip(a, b)} for a, b in zip(topk_txt, topk_scores)]
    except IndexError:
        topk_all = [{}]
    return topk_all[0]


def encode_question(question: str, textual_description="") -> Dict[str, torch.Tensor]:
    """
    Encodes a natural language question as a tokenized input suitable for input
    to a transformer model. The encoding includes special tokens to mark the beginning
    and end of the input, as well as a mask token to indicate where the answer
    should be predicted.
    """
    assert qa_model.tokenizer is not None, "Tokenizer not loaded"

    # Capitalize and strip whitespace from the question string
    question = question.capitalize().strip()

    # If the question contains a [MASK] token, replace it with the mask token
    if "[MASK]" in question:
        question = question.replace("[MASK]", qa_model.tokenizer.mask_token)
        text = f"{args.prefix} {question}{args.suffix}"
    # Otherwise, add "Question: " and "Answer: " tags to the question and mask the answer
    else:
        if question[-1] != "?":
            question = str(question) + "?"
        text = f"{args.prefix} Question: {question} Answer: {qa_model.tokenizer.mask_token}. Subtitles: {textual_description} {args.suffix}"

    # Tokenize the text and encode the resulting token ids as a PyTorch tensor
    encoded = qa_model.tokenizer(
        [text],
        add_special_tokens=True,
        max_length=args.max_tokens,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    return encoded


# Get general textual description for a scene
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


# Get answers from a list of images (randomly selected)
async def get_answers_from_images(images: List[str], question: str) -> List[str]:
    answers = []
    # Filter out empty string images
    images = [image for image in images if image]
    print(images[0])

    # Build a scene from list of images
    first_image = BASIC_DICT[images[0]]
    event = Event(
        start_time=BASIC_DICT[images[0]]["time"],
        end_time=BASIC_DICT[images[-1]]["time"],
        location=first_image["location"],
        location_info=first_image["location_info"],
        country=first_image["country"],
        region=first_image["region"],
        ocr=[],
    )
    for image in images:
        for text in BASIC_DICT[image]["ocr"]:
            if text and text not in event.ocr:
                event.ocr.append(text)

    # Get textual description for a scene
    textual_description = get_specific_description(event)
    # Get answers from textual description
    QA_input = {"question": question, "context": textual_description}
    prompt = QA_PROMPT.format(question=question, events=textual_description)

    text_answers = await llm_model.generate_from_text(prompt)

    # Get answers from images
    encoded_question = encode_question(question)
    video_answers = answer(images, encoded_question)

    # Combine answers from text and images
    answers.extend(list(video_answers.keys()))

    return answers


async def answer_text_only(question: str, events: EventResults, k: int = 10):
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

    return answers
