import json
import os
from .utils import *

TIMELINE_SPAN = 9  # If they want more, submit more
group_info = json.load(open(f"{COMMON_PATH}/group_info.json"))
scene_info = json.load(open(f"{COMMON_PATH}/scene_info.json"))

def get_next_scenes(date, group_range, max_scene):
    scenes = []
    for index in group_range:
        new_group_id = f"{date}_{index}"
        if new_group_id in group_info[date]:
            for scene in group_info[date][new_group_id]:
                if int(scene.split('_')[-1]) >= max_scene:
                    scenes.append(group_info[date][new_group_id][scene])
                    if len(scenes) > TIMELINE_SPAN:
                        return scenes, -1
    return scenes, -1


def get_prev_scenes(date, group_range, min_scene):
    scenes = []
    group_range = list(group_range)[::-1]
    for index in group_range:
        new_group_id = f"{date}_{index}"
        if new_group_id in group_info[date]:
            for scene in group_info[date][new_group_id]:
                if int(scene.split('_')[-1]) <= min_scene:
                    scenes.append(group_info[date][new_group_id][scene])
                    if len(scenes) > TIMELINE_SPAN:
                        return scenes, -1
    return scenes, -1

def get_timeline(images, timeline_type="after", direction="next"):
    images = [grouped_info_dict[image]for image in images]
    scene_id = int(images[0]["scene"].split('_')[-1])
    group_id = int(images[0]["group"].split('_')[-1])
    scenes = []
    date = images[0]["scene"].split("/")[0]
    marked = -1

    if timeline_type == "after":
        group_id = int(images[0]["group"].split('_')[-1])
        if f"{date}_{group_id + 1}" in group_info[date]:
            group_id += 1
            next_scene_id = int(
                list(group_info[date][f"{date}_{group_id}"].keys())[0].split('_')[-1])
        else: # At the end
            next_scene_id = int(
                list(group_info[date][f"{date}_{group_id}"].keys())[-1].split('_')[-1])
        scene_id = next_scene_id

    elif timeline_type == "before":
        group_id = int(images[0]["group"].split('_')[-1])
        if f"{date}_{group_id - 1}" in group_info[date]:
            group_id -= 1
        prev_scene_id = int(list(
            group_info[date][f"{date}_{group_id}"].keys())[0].split('_')[-1])
        scene_id = prev_scene_id

    for index in range(scene_id-10, scene_id + 10):
            scene = f"{date}/scene_{index}"
            if scene in scene_info[date]:
                if f"{date}/scene_{scene_id}" == scene:
                    marked = len(scenes)
                scenes.append(scene_info[date][scene])

    return scenes, marked, group_id

def get_timeline_group(date):
    groups = []
    for group in group_info[date]:
        scene = list(group_info[date][group].items())[0][0]
        groups.append(scene_info[date][scene][0])
    return groups


def get_full_scene_from_image(image, group_factor='group'):
    assert group_factor in [
        "group", "scene"], f"Invalid value of group_factor({group_factor}). Use \"scene\" or \"group\"."
    group_id = grouped_info_dict[image]["group"]
    date = group_id.split("_")[0]
    if group_factor == "group":
        return [img for scene in group_info[date][group_id]
                for img in group_info[date][group_id][scene]]
    else:
        scene_id = grouped_info_dict[image]["scene"]
        return [img for img in group_info[date][group_id][scene_id]]


def get_multiple_scenes_from_images(begin_image, end_image, group_factor='group'):
    min_group = grouped_info_dict[begin_image]["group"]
    max_group = grouped_info_dict[end_image]["group"]
    date = min_group.split("_")[0]
    min_group = int(min_group.split("_")[1])
    max_group = int(max_group.split("_")[1])
    group_range = range(min_group, max_group + 1)

    if group_factor == "group":
        images = []
        for index in group_range:
            new_group_id = f"{date}_{index}"
            if new_group_id in group_info[date]:
                images.extend([img for scene in group_info[date][new_group_id]
                               for img in group_info[date][new_group_id][scene]])
    else:
        min_scene = grouped_info_dict[begin_image]["scene"]
        max_scene = grouped_info_dict[end_image]["scene"]
        min_scene = int(min_scene.split("_")[1])
        max_scene = int(max_scene.split("_")[1])
        images = []
        scene_range = range(min_scene, max_scene + 1)
        for index in group_range:
            new_group_id = f"{date}_{index}"
            if new_group_id in group_info[date]:
                images.extend([img for scene in group_info[date][new_group_id]
                               for img in group_info[date][new_group_id][scene] if int(scene.split('_')[1]) in scene_range])
    return images


# # For ARRON, use something like this
# print(get_full_scene_from_image(
#     "2016-08-15/20160815_053701_000.jpg", group_factor="group"))
# print(get_multiple_scenes_from_images("2016-08-23/20160823_122201_000.jpg",
#                                       "2016-08-23/20160823_122201_000.jpg", group_factor="group"))
