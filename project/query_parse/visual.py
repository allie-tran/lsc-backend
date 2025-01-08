import os
from typing import List, Tuple

import numpy as np
import open_clip
import pandas as pd
import torch
from configs import (
    CLIP_EMBEDDINGS,
    DATA_YEARS,
    EMBEDDING_DIM,
    IMAGE_DIRECTORY,
    MODEL_NAME,
    PRETRAINED_DATASET,
)
from numpy import linalg as LA
from open_clip.model import CLIP
from open_clip.tokenizer import _tokenizer
from PIL import Image as PILImage
from query_parse.types.requests import Data
from results.models import Image
from transformers import AutoModel, AutoProcessor

from .constants import DESCRIPTIONS
from .types.elasticsearch import VisualInfo
from .utils import search_keywords

# Load CLIP Model
device = "cpu"
# if not FORCE_CPU and torch.cuda.is_available():  # type: ignore
#     device = "cuda"


def load_features(paths):
    # Load pre-embedded photo features
    photo_features = np.zeros((0, EMBEDDING_DIM))
    photo_ids = []
    if not paths:
        return photo_features, {}, []

    for path in paths:
        print(path)
        # For photo features
        # new_photo_features = np.load(
        #     f"{CLIP_EMBEDDINGS}/{year}/{MODEL_NAME}_{PRETRAINED_DATASET}_nonorm/features.npy"
        # )
        if not os.path.exists(f"{path}/features.npy"):
            continue
        new_photo_features = np.load(f"{path}/features.npy")
        if photo_features.size == 0:
            photo_features = new_photo_features
        else:
            photo_features = np.concatenate([photo_features, new_photo_features])

        # For photo ids
        # new_photo_ids = pd.read_csv(
        #     f"{CLIP_EMBEDDINGS}/{year}/{MODEL_NAME}_{PRETRAINED_DATASET}_nonorm/photo_ids.csv"
        # )
        new_photo_ids = pd.read_csv(f"{path}/photo_ids.csv")
        if isinstance(new_photo_ids, pd.DataFrame):
            new_photo_ids = new_photo_ids["photo_id"]
            if new_photo_ids is not None:
                photo_ids = np.concatenate([photo_ids, new_photo_ids])

    # Add .jpg extension to photo ids if not present
    photo_ids = [
        photo_id + ".jpg" if "." not in photo_id else photo_id for photo_id in photo_ids
    ]

    # Normalize photo features
    norm_photo_features = photo_features / LA.norm(
        photo_features, keepdims=True, axis=-1
    )
    image_to_id = {image: i for i, image in enumerate(photo_ids)}

    return norm_photo_features, image_to_id, photo_ids


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


class ClipModel:
    def __init__(self):
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED_DATASET, device=device
        )
        assert isinstance(clip_model, CLIP), "Model is not CLIP"
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        self.clip_model = clip_model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # LSC23 dataset
        lsc_paths = [
            f"{CLIP_EMBEDDINGS}/{year}/ViT-L-14-336_openai_nonorm"
            for year in DATA_YEARS
        ]

        # Deakin dataset (None for now)
        deakin_paths = []

        # Both datasets
        self.combine_datasets(lsc_paths, deakin_paths)

    def combine_datasets(self, lsc_paths, deakin_paths):
        lsc_norm_photo_features, lsc_image_to_id, lsc_photo_ids = load_features(lsc_paths)
        deakin_norm_photo_features, deakin_image_to_id, deakin_photo_ids = load_features(deakin_paths)

        # Both datasets
        self.norm_photo_features = {
            Data.LSC23: lsc_norm_photo_features,
            Data.Deakin: deakin_norm_photo_features,
        }

        self.image_to_id = {
            Data.LSC23: lsc_image_to_id,
            Data.Deakin: deakin_image_to_id,
        }

        self.photo_ids = {
            Data.LSC23: lsc_photo_ids,
            Data.Deakin: deakin_photo_ids,
        }

    def encode_text(self, main_query: str) -> np.ndarray:
        with torch.no_grad():
            sentences = _split_text(main_query, 77)
            tokens = self.tokenizer(sentences).to(device)
            text_encoded = self.clip_model.encode_text(tokens)  # type: ignore

            if len(sentences) > 1:
                text_encoded = text_encoded.mean(dim=0, keepdim=True)  # type: ignore
            else:
                text_encoded = text_encoded.squeeze(0)
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)  # type: ignore

        text_features = text_encoded.cpu().numpy()
        return text_features

    def encode_image(self, image_path: str) -> np.ndarray:
        image_read = PILImage.open(f"{IMAGE_DIRECTORY}/{image_path}")
        image_tensor = self.preprocess(image_read).unsqueeze(0).to(device)  # type: ignore
        with torch.no_grad():
            image_feat = self.clip_model.encode_image(image_tensor)
            image_feat /= image_feat.norm(dim=-1, keepdim=True)
        return image_feat.cpu().numpy()

    def score_images(
        self, image_objs: List[Image], encoded_query: np.ndarray, data: Data
    ) -> List[float]:
        images = [image.src for image in image_objs]
        try:
            encoded_query /= LA.norm(encoded_query, keepdims=True, axis=-1)
        except TypeError:
            return [0 for _ in images]
        if images:
            image_features = self.norm_photo_features[data][
                np.array([self.image_to_id[data][image] for image in images])
            ]
            similarity = image_features @ encoded_query.T  # B x D @ D x 1 = B x 1
            similarity = similarity.reshape(-1)
            similarity[np.where(similarity < 0.1)] = 0
            return similarity.astype("float").tolist()
        return []

    def score_all_images(self, query: str, data: Data) -> Tuple[List[float], np.ndarray]:
        encoded_query = self.encode_text(query)
        try:
            encoded_query /= LA.norm(encoded_query, keepdims=True, axis=-1)
        except TypeError:
            return [0.0 for _ in self.photo_ids[data]], encoded_query
        similarity = self.norm_photo_features[data] @ encoded_query.T
        similarity = similarity.reshape(-1)
        similarity[np.where(similarity < 0.1)] = 0
        return similarity.astype("float").tolist(), encoded_query


class SIGLIP(ClipModel):
    def __init__(self):
        model = AutoModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384",
            device_map=device,
        )
        self.model = model
        self.processor = processor

        # LSC23 dataset
        lsc_paths = [
                f"{CLIP_EMBEDDINGS}/{year}/google-siglip-so400m-patch14-384_nonorm"
                for year in DATA_YEARS
            ]

        # Deakin dataset
        deakin_paths = [f"{CLIP_EMBEDDINGS}/Deakin/google-siglip-so400m-patch14-384"]

        # Both datasets
        self.combine_datasets(lsc_paths, deakin_paths)


    def encode_text(self, main_query: str) -> np.ndarray:
        inputs = self.processor(
            text=[main_query], return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs).mean(dim=0)
        return outputs.cpu().numpy().astype("float")

    def encode_image(self, image_path: str) -> np.ndarray:
        image_read = PILImage.open(f"{IMAGE_DIRECTORY}/{image_path}")
        photo_preprocessed = self.processor(images=image_read, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**photo_preprocessed)
        return outputs.cpu().numpy().astype("float")


# This is for Deakin dataset
class CLIPA(ClipModel):
    def __init__(
        self, model_name: str = "hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B"
    ):
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, device=device
        )

        assert isinstance(clip_model, CLIP), "Model is not CLIP"
        tokenizer = open_clip.get_tokenizer(model_name)

        self.clip_model = clip_model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # LSC23 dataset
        lsc_paths = []

        # Deakin dataset
        deakin_paths = [f"{CLIP_EMBEDDINGS}/Deakin/clipa"]

        # Both datasets
        self.combine_datasets(lsc_paths, deakin_paths)


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


clip_model = ClipModel()
siglip_model = SIGLIP()
clipa_model = CLIPA()

def get_model(data: Data):
    if data == Data.LSC23:
        return siglip_model
    elif data == Data.Deakin:
        return clipa_model

def encode_text(*args, chosen_model: ClipModel = siglip_model, **kwargs):
    return chosen_model.encode_text(*args, **kwargs)

def encode_image(*args, chosen_model: ClipModel = siglip_model, **kwargs):
    return chosen_model.encode_image(*args, **kwargs)

def score_images(*args, chosen_model: ClipModel = siglip_model, **kwargs):
    return chosen_model.score_images(*args, **kwargs)

def norm_photo_features(data: Data):
    chosen_model = get_model(data)
    return chosen_model.norm_photo_features[data]

def photo_ids(data: Data):
    chosen_model = get_model(data)
    return chosen_model.photo_ids[data]
