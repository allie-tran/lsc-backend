from typing import List

import numpy as np
import open_clip
import pandas as pd
import torch
from configs import (
    CLIP_EMBEDDINGS,
    DATA_YEARS,
    EMBEDDING_DIM,
    FORCE_CPU,
    MODEL_NAME,
    PRETRAINED_DATASET,
)
from numpy import linalg as LA
from open_clip.model import CLIP
from open_clip.tokenizer import _tokenizer

from .constants import DESCRIPTIONS
from .types import VisualInfo
from .utils import search_keywords

# Load CLIP Model
device = "cpu"
if not FORCE_CPU and torch.cuda.is_available():  # type: ignore
    device = "cuda"

# Load pre-embedded photo features
photo_features = np.zeros((0, EMBEDDING_DIM))
photo_ids = []

for year in DATA_YEARS:
    # For photo features
    new_photo_features = np.load(
        f"{CLIP_EMBEDDINGS}/{year}/{MODEL_NAME}_{PRETRAINED_DATASET}_nonorm/features.npy"
    )
    photo_features = np.concatenate([photo_features, new_photo_features])

    # For photo ids
    new_photo_ids = pd.read_csv(
        f"{CLIP_EMBEDDINGS}/{year}/{MODEL_NAME}_{PRETRAINED_DATASET}_nonorm/photo_ids.csv"
    )
    if isinstance(new_photo_ids, pd.DataFrame):
        new_photo_ids = new_photo_ids["photo_id"]
        if new_photo_ids is not None:
            photo_ids = np.concatenate([photo_ids, new_photo_ids])

# Add .jpg extension to photo ids if not present
photo_ids = [
    photo_id + ".jpg" if "." not in photo_id else photo_id for photo_id in photo_ids
]

# Normalize photo features
norm_photo_features = photo_features / LA.norm(photo_features, keepdims=True, axis=-1)
image_to_id = {image: i for i, image in enumerate(photo_ids)}

# Load CLIP model
clip_model, *_ = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED_DATASET, device=device
)
assert isinstance(clip_model, CLIP), "Model is not CLIP"
tokenizer = open_clip.get_tokenizer(MODEL_NAME)


# Detect if the tokenized text is longer than the context length
def _check_context_length(text: str, context_length: int) -> bool:
    tokens = _tokenizer.encode(text)
    if len(tokens) > context_length:
        return False
    return True


# If the tokenized text is longer than the context length, split it into multiple sentences
def _split_text(text: str, context_length: int) -> List[str]:
    sentences = text.split(".")
    result = []
    while sentences:
        sentence = sentences.pop(0)
        while sentences and _check_context_length(
            sentence + "." + sentences[0], context_length
        ):
            sentence += "." + sentences.pop(0)
        result.append(sentence)
    return result


def encode_text(main_query: str) -> np.ndarray:
    with torch.no_grad():
        sentences = _split_text(main_query, 77)
        tokens = tokenizer(sentences).to(device)
        text_encoded = clip_model.encode_text(tokens)

        if len(sentences) > 1:
            print("multiple sentences")
            print(sentences)
            text_encoded = text_encoded.mean(dim=0, keepdim=True)

        # text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    text_features = text_encoded.cpu().numpy()
    return text_features


def score_images(images: List[str], encoded_query: np.ndarray) -> List[float]:
    try:
        encoded_query /= LA.norm(encoded_query, keepdims=True, axis=-1)
    except TypeError as e:
        return [0 for _ in images]
    if images:
        image_features = norm_photo_features[
            np.array([image_to_id[image] for image in images])
        ]
        similarity = image_features @ encoded_query.T  # B x D @ D x 1 = B x 1
        similarity = similarity.reshape(-1)
        similarity[np.where(similarity < 0.1)] = 0
        return similarity.astype("float").tolist()
    return []


def search_for_visual(text: str) -> VisualInfo:
    """
    Search for visual information
    """
    concepts = search_keywords(DESCRIPTIONS, text)
    visual_info = VisualInfo(
        text=text,
        concepts=concepts,
    )
    return visual_info
