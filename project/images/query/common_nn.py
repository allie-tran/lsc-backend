import pandas as pd
import numpy as np
from ..nlp_utils.common import FILES_DIRECTORY
import open_clip
import torch
from scipy.special import softmax
import os
from numpy import linalg as LA

CLIP_EMBEDDINGS = os.environ.get("CLIP_EMBEDDINGS")
PRETRAINED_MODELS = os.environ.get("PRETRAINED_MODELS")

# CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# model_name = "ViT-H-14"
# pretrained = "laion2b_s32b_b79k"
model_name = "ViT-L-14-336"
pretrained = "openai"

def get_embeddings(year):
    CLIP_EMBEDDINGS = f"{os.getenv('CLIP_EMBEDDINGS')}/{year}/"
    photo_features = np.load(f"{CLIP_EMBEDDINGS}/{model_name}_{pretrained}_nonorm/features.npy")
    photo_ids = pd.read_csv(f"{CLIP_EMBEDDINGS}/{model_name}_{pretrained}_nonorm/photo_ids.csv")["photo_id"].to_list()
    photo_ids = [photo_id + '.jpg' if '.' not in photo_id else photo_id for photo_id in photo_ids]
    return photo_features, photo_ids

photo_features, photo_ids = get_embeddings('LSC20')
new_photo_features, new_photo_ids = get_embeddings('LSC23')
photo_features = np.concatenate((photo_features, new_photo_features))
photo_ids = photo_ids + new_photo_ids
new_photo_features = None
# photo_features = np.load(f"{FILES_DIRECTORY}/embeddings/features.npy")
# photo_ids = pd.read_csv(f"{FILES_DIRECTORY}/embeddings/photo_ids.csv")["photo_id"].to_list()
norm_photo_features = photo_features / LA.norm(photo_features, keepdims=True, axis=-1)
DIM = photo_features[0].shape[-1]
image_to_id = {image: i for i, image in enumerate(photo_ids)}

clip_model, *_ = open_clip.create_model_and_transforms(model_name, 
                                                                 pretrained=pretrained,
                                                                 device=device)
tokenizer = open_clip.get_tokenizer(model_name)
from open_clip.tokenizer import _tokenizer
# Detect if the tokenized text is longer than the context length
def _check_context_length(text: str, context_length: int) -> bool:
    tokens = _tokenizer.encode(text)
    if len(tokens) > context_length:
        return False
    return True

# If the tokenized text is longer than the context length, split it into multiple sentences
def _split_text(text: str, context_length: int):
    sentences = text.split(".")
    result = []
    while sentences:
        sentence = sentences.pop(0)
        while sentences and _check_context_length(sentence + "." + sentences[0], context_length):
            sentence += "." + sentences.pop(0)
        result.append(sentence)
    return result            

def encode_query(main_query):
    with torch.no_grad():
        sentences = _split_text(main_query, 77)
        main_query = tokenizer(sentences).to(device)
        text_encoded = clip_model.encode_text(main_query)
        
        if len(sentences) > 1:
            print("multiple sentences")
            print(sentences)
            text_encoded = text_encoded.mean(dim=0, keepdim=True)
        
        # text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    text_features = text_encoded.cpu().numpy()
    return text_features

def score_images(images, encoded_query):
    try:
        encoded_query /= LA.norm(encoded_query, keepdims=True, axis=-1)
    except TypeError as e:
        return [0 for _ in images]
    if images:
        # print(np.array([image_to_id[image] for image in images]))
        image_features = norm_photo_features[np.array([image_to_id[image] for image in images])]
        similarity = image_features @ encoded_query.T # B x D @ D x 1 = B x 1
        similarity = similarity.reshape(-1)
        similarity[np.where(similarity < 0.1)] = 0
        return similarity.astype("float").tolist()
    return []



