import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from configs import (
    BUILD_ON_STARTUP,
    MSRVTT_VOCAB_PATH,
    QA_DIM,
    QA_FEATURES,
    QA_IDS,
    QA_LLM_BASE,
    QA_PATH,
    TEXT_QA_MODEL,
)
from query_parse.constants import BASIC_DICT
from query_parse.visual import device
from results.models import Event
from transformers.pipelines import pipeline
from transformers.utils import logging as transformers_logging

from .FrozenBiLM.args import get_args_parser
from .FrozenBiLM.model import build_model, get_tokenizer
from .FrozenBiLM.util.misc import get_mask

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

nlp = None
visual_qa = None
id2a = {}
tokenizer = None

# Get FrozenBiLM arguments
parser = argparse.ArgumentParser(parents=[get_args_parser()])
args = parser.parse_args(
    f"""--combine_datasets msrvtt --combine_datasets_val msrvtt \
--suffix="." --max_tokens=256 --ds_factor_ff=8 --ds_factor_attn=8 \
--load={QA_PATH} \
--msrvtt_vocab_path={MSRVTT_VOCAB_PATH},
--model_name {QA_LLM_BASE}""".split()
)
if args.save_dir:
    args.save_dir = os.path.join(args.presave_dir, args.save_dir)


def get_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load the vocabulary for the MSRVTT dataset and return a dictionary mapping
    words to their corresponding token ids.
    """
    global id2a
    # vocab = json.load(open(args.msrvtt_vocab_path, "r"))
    vocab = pd.read_csv(VOCAB_PATH)["0"].to_list()[:3000]  # type: ignore
    vocab = [a for a in vocab if a != np.nan and str(a) != "nan"]
    vocab = {a: i for i, a in enumerate(vocab)}
    id2a = {y: x for x, y in vocab.items()}
    return vocab, id2a


def build_qa_model() -> None:
    """
    Build a question-answering model using the provided arguments and load a pretrained
    checkpoint if available.
    """
    global visual_qa
    global tokenizer
    global id2a
    global nlp
    nlp = pipeline(
        "question-answering",
        model=TEXT_QA_MODEL,
        tokenizer=TEXT_QA_MODEL,
        topk=5,
        device=-1,
        truncation=True,
        padding=True,
    )
    # Build model
    print("Building QA model")
    tokenizer = get_tokenizer(args)
    args.n_ans = 2
    visual_qa = build_model(args)
    assert isinstance(visual_qa, torch.nn.Module), "Model is not a torch.nn.Module"
    visual_qa.to(device)
    visual_qa.eval()

    # Load pretrained checkpoint
    assert args.load
    print("Loading from", args.load)
    checkpoint = torch.load(args.load, map_location="cpu")
    visual_qa.load_state_dict(checkpoint["model"], strict=False)

    # Init answer embedding module
    vocab, id2a = get_vocab()
    aid2tokid = torch.zeros(len(vocab), args.max_atokens).long()
    for a, aid in vocab.items():
        try:
            tok = torch.tensor(
                tokenizer(
                    a,
                    add_special_tokens=False,
                    max_length=args.max_atokens,
                    truncation=True,
                    padding="max_length",
                )["input_ids"],
                dtype=torch.long,
            )
            aid2tokid[aid] = tok
        except ValueError as e:
            print(a, aid)
            raise (e)
    visual_qa.set_answer_embeddings(aid2tokid.to(device), freeze_last=args.freeze_last)


if BUILD_ON_STARTUP:
    build_qa_model()


def answer(
    images: List[str], encoded_question: Dict[str, torch.Tensor]
) -> Dict[str, str]:
    """
    Given a list of image filenames and an encoded question, generates the top 5 answers
    to the question based on the provided image embeddings.
    """
    assert visual_qa is not None, "Model not loaded"
    assert isinstance(visual_qa, torch.nn.Module), "Model is not a torch.nn.Module"
    assert tokenizer is not None, "Tokenizer not loaded"
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
        attention_mask[input_ids == tokenizer.sep_token_id] = 0
        input_ids[input_ids == tokenizer.sep_token_id] = tokenizer.pad_token_id

    # Use the provided BiLM to generate a prediction for the mask token
    output = visual_qa(
        video=video,
        video_mask=video_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Extract the logits for the mask token
    logits = output["logits"]
    delay = args.max_feats if args.use_video else 0
    logits = logits[:, delay : encoded_question["input_ids"].size(1) + delay][
        encoded_question["input_ids"] == tokenizer.mask_token_id
    ]  # get the prediction on the mask token
    logits = logits.softmax(-1)
    try:
        topk = torch.topk(logits, 5, -1)
        topk_txt = [[id2a[int(x.item())] for x in y] for y in topk.indices.cpu()]
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
    assert tokenizer is not None, "Tokenizer not loaded"

    # Capitalize and strip whitespace from the question string
    question = question.capitalize().strip()

    # If the question contains a [MASK] token, replace it with the mask token
    if "[MASK]" in question:
        question = question.replace("[MASK]", tokenizer.mask_token)
        text = f"{args.prefix} {question}{args.suffix}"
    # Otherwise, add "Question: " and "Answer: " tags to the question and mask the answer
    else:
        if question[-1] != "?":
            question = str(question) + "?"
        text = f"{args.prefix} Question: {question} Answer: {tokenizer.mask_token}. Subtitles: {textual_description} {args.suffix}"

    # Tokenize the text and encode the resulting token ids as a PyTorch tensor
    encoded = tokenizer(
        [text],
        add_special_tokens=True,
        max_length=args.max_tokens,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )

    return encoded


# Get textual description for a scene
def get_textual_description(event: Event, question: str) -> str:
    if nlp is None:
        build_qa_model()

    # converting datetime to string
    start_time = event.start_time.strftime("%I:%M%p")
    end_time = event.end_time.strftime("%I:%M%p")

    # calculating duration
    time_delta = event.end_time - event.start_time
    if time_delta.seconds > 0:
        hours = time_delta.seconds // 3600
        if time_delta.days > 0:
            if hours > 0:
                duration = f"{time_delta.days} days and {hours} hours"
            else:
                duration = f"{time_delta.days} days"
        else:
            if hours > 0:
                minutes = (time_delta.seconds - hours * 3600) // 60
                duration = f"{hours} hours and {minutes} minutes"
            elif time_delta.seconds < 60:
                duration = f"{time_delta.seconds} seconds"
            else:
                minutes = time_delta.seconds // 60
                duration = f"{minutes} minutes"
        duration = f", lasted for about {duration}"
        time = f"from {start_time} to {end_time}"
    else:
        duration = ""
        time = f"at {start_time}"

    # converting datetime to string like Jan 1, 2020
    date = event.start_time.strftime("%A, %B %d, %Y")

    location = ""
    location_info = ""
    if event.original_location != "---":
        location = f"in {event.original_location}"
        if event.location_info:
            location_info = f", which is a {event.location_info}"
    ocr = ""
    if event.ocr:
        ocr = f"Some texts that can be seen from the images are: {' '.join(event.ocr)}."

    regions = [reg for reg in event.region if reg != event.country]

    textual_description = (
        f"The event happened {time}{duration} on {date} "
        + f"{location}{location_info} in {', '.join(regions)} in {event.country}. {ocr}"
    )

    return textual_description


# Get answers from a list of images (randomly selected)
def get_answers_from_images(images: List[str], question: str) -> List[str]:
    assert nlp is not None, "Model not loaded"
    answers = []
    # Filter out empty string images
    images = [image for image in images if image]
    print(images[0])

    # Build a scene from list of images
    first_image = BASIC_DICT[images[0]]
    event = Event(
        start_time=BASIC_DICT[images[0]]["time"],
        end_time=BASIC_DICT[images[-1]]["time"],
        original_location=first_image["location"],
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
    textual_description = get_textual_description(event, question)
    # Get answers from textual description
    QA_input = {"question": question, "context": textual_description}
    results = nlp(QA_input)
    for res in results:
        if res and res["score"] > 0.1:  # type: ignore
            answers.append(res["answer"])  # type: ignore

    # Get answers from images
    encoded_question = encode_question(question)
    answers.extend(list(answer(images, encoded_question).keys()))

    return answers


def answer_topk_scenes(
    question: str, events: List[Event], scores: List[float], k: int = 10
):
    """
    Given a natural language question and a list of scenes, returns the top k answers
    to the question across all scenes. Uses an encoding of the question and an answer
    function to compute answer scores for each scene.

    Args:
    - question (str): a natural language question
    - scenes (list of dicts): a list of scenes, each represented as a dictionary with
      the following keys:
        - "current" (list of tuples): a list of (image, score) with for the scene
    - k (int): the number of top scenes to consider

    Returns:
    - answers (list of str): the top 10 answers to the question across top-k scenes
    """
    # Create a defaultdict to accumulate answer scores across all scenes
    answers = defaultdict(float)
    if visual_qa is None:
        build_qa_model()

    assert nlp is not None, "Model not loaded"

    # Encode the question using a helper function
    original_question = question

    # Iterate over each scene
    for i, (event, score) in enumerate(zip(events[:k], scores[:k])):
        # Extract the images from the "current" field of the scene
        images = [i[0] for i in event.images if i[0]]

        # Extract textual information from the scene
        textual_description = ""
        for image in images:
            for text in BASIC_DICT[image]["ocr"]:
                if text and text not in event.ocr:
                    event.ocr.append(text)
        try:
            textual_description = get_textual_description(event, question)
            QA_input = {"question": original_question, "context": textual_description}
            results = nlp(QA_input)
            for res in results:
                if res["score"] > 0.1:  # type: ignore
                    ans_score = res["score"] * (10 - i) ** 2 / 5  # type: ignore
                    answers[res["answer"]] = max(answers[res["answer"]], ans_score)  # type: ignore
        except Exception as e:
            print("Error in TextQA", e)
            pass

        encoded_question = encode_question(question, textual_description)

        # Compute answer scores for the current scene using an answer function
        ans = answer(images, encoded_question)

        # Accumulate the answer scores in the defaultdict
        for a, s in ans.items():
            answers[a] += float(s)

    # Sort the answers by score and take the top 10, discarding one if zero scores
    answers = [
        a for a, s in sorted(answers.items(), key=lambda x: x[1], reverse=True) if s > 0
    ][:10]

    return answers
