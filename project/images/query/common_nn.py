import pandas as pd
import numpy as np
# import clip
import open_clip
import torch
from scipy.special import softmax
import os
from numpy import linalg as LA

# CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model_name = "ViT-H-14"
pretrained = "laion2b_s32b_b79k"
DIM = 1024
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


