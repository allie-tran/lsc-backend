import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from configs import QA_DIM, QA_FEATURES, QA_IDS
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


def videoqa_answer(
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
