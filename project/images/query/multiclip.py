import torch.nn as nn
from lifelog_qa.multiclip import *
from .common_nn import *
from .utils import *
import clip
import torch

import os
import torch
from tqdm.auto import tqdm
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

MODEL_VER = StraightCLIP
save_file = ""
device = "cuda"
train_config = {}
if os.path.exists(save_file):
    states = torch.load(save_file, map_location="cpu")
    if "config" in states:
        print("="*80)
        print("Config")
        print(states["config"])
        train_config = states["config"]
        print("="*80)
        print("Info")
        print(states["info"])
else:
    train_config = {
        'model_name':  "ViT-L/14@336px",
        "output_weights": False,
        "encoded": True,
        "clip_path": "",
        "alpha": 0.0,
        "n_cluster": 3
    }

if "clip_path" not in train_config:
    train_config["clip_path"] = ""
#=======================#
use_full_text = check_if_use_full_text(MODEL_VER)

# Pretty much unchanged
scene_agg, preprocess, *_ = load_scene_model(save_file, MODEL_VER, train_config,
                                             device, train=False)
scene_agg.clip = clip_model
# scene_agg.normed_photo_features = photo_features
# scene_agg.prepared = True

if device != "cpu":
    convert_models_to_fp16(scene_agg)
else:
    convert_models_to_fp32(scene_agg)

scenes = list(scene_segments.values())
scene_ids = list(scene_segments.keys())
new_scenes = []
for scene in tqdm(scenes, desc="converting image to id"):
    new_scene = [image_to_id[image] for image in scene if image in image_to_id]
    new_scenes.append(new_scene)
scenes = new_scenes

expanded_scenes = []
expanded_scene_ids = []
scene_structures = []
new_scenes = []

for i, scene in tqdm(enumerate(scenes), total=len(scenes)):
    # if targets.intersection(scene):
    expanded_list = []
    for r in range(0, 2):
        if i >= r:
            expanded = range(i, min(len(scenes), i + 1 + r))
            images = [image for j in expanded for image in scenes[j]]
            if images:
                expanded_list.append(len(expanded_scenes))
                expanded_scenes.append(images)
                expanded_scene_ids.append([scene_ids[j] for j in expanded])
    scene_structures.append(expanded_list)
    new_scenes.append(scene)
scenes = new_scenes

# groups = json.load(open(f"{FILES_DIRECTORY}/group_segments.json"))
# scene_segments = {}
# for group_name in groups:
#     for scene_name, images in groups[group_name]["scenes"]:
#         assert "S_" in scene_name, f"{scene_name} is not a valid scene id"
#         scene_segments[scene_name] = images

def query_scene(scene_agg, main_query, size=100, flatten=False, choose_best=False, progress=None):
    global photo_features
    with torch.no_grad():
        tokens = clip.tokenize(main_query).to(device)
        if use_full_text:
            text_encoded, text_hidden = scene_agg.encode_text(tokens)
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        else:
            text_encoded = scene_agg.encode_text(tokens)
    
    text_features = text_encoded.cpu().numpy()
    print("prepare_inference")
    photo_features, text_features = scene_agg.prepare_inference(photo_features=photo_features, #type: ignore
                                                                text_features=text_features,
                                                                expanded_scenes=expanded_scenes)
    print("inferencing")
    if use_full_text:
        similarities, best_candidates = scene_agg.inference(
            (text_features, text_hidden), expanded_scenes, photo_features, progress=progress)  # type: ignore
    else:
        similarities, best_candidates = scene_agg.inference(text_features, expanded_scenes, photo_features, progress=progress)
    assert len(similarities) == len(expanded_scenes), f"wrong length {len(similarities)} != {len(expanded_scenes)}"
    # Aggregate scores based on structure:
    scores = [0 for i in range(len(scenes))]
    ranges = [[] for i in range(len(scenes))]
    
    for i, expanded_list in enumerate(scene_structures):
        sims = [similarities[j] for j in expanded_list]
        max_ind = np.argmax(sims)
        try:
            scores[i] = sims[max_ind]
            ranges[i] = expanded_list[max_ind]
        except:
            print(expanded_list, max_ind)
            print(sims)
    
    # scores = similarities
    best_scenes = sorted(zip(scores, range(len(
        scores))), key=lambda x: x[0], reverse=True)
    idx = list(zip(*best_scenes))[1]
    
    results = []
    result_weights = []
    for i in idx:
        expanded_scene_idx = ranges[i]
        scene = expanded_scenes[expanded_scene_idx]  # type: ignore
        best_idx, best_val = zip(*best_candidates[expanded_scene_idx]) # type: ignore
        weights = {photo_ids[scene[j]]: float(best_val[j])
                   for j in best_idx if best_val[j] > 0.1}
        best_images = [photo_ids[scene[j]] for j in best_idx if best_val[j] > 0.1]
        scene = best_images
        results.append((expanded_scene_ids[expanded_scene_idx], scene))
        result_weights.append(weights)
        if len(results) >= size:
            break
    return results, result_weights


def get_relevance_scenes(query):
    results, weights = query_scene(
        scene_agg, query, size=100, choose_best=False, progress=tqdm)
    return results, weights
