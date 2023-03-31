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
device = "cpu"
# clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)
model_name = "ViT-H-14"
pretrained = "laion2b_s32b_b79k"
DIM = 1024
clip_model, *_ = open_clip.create_model_and_transforms(model_name, 
                                                       pretrained=pretrained,
                                                       device=device)
tokenizer = open_clip.get_tokenizer(model_name)

def encode_query(main_query):
    with torch.no_grad():
        main_query = tokenizer([main_query]).to(device)
        text_encoded = clip_model.encode_text(main_query)
        # text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    text_features = text_encoded.cpu().numpy()
    return text_features


