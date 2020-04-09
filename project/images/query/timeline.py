import json
import os
from .utils import *

TIMELINE_SPAN = 10  # If they want more, submit more

def get_timeline(images, timeline_type="after"):
    images = [grouped_info_dict[image]for image in images]
    group_ids = [int(image["group"].split('_')[-1]) for image in images]
    print(timeline_type)
    scenes = []
    if timeline_type == "current":
        min_group = min(group_ids)
        max_group = max(group_ids)
        group_range = range(min_group, max_group + 1)

        date = images[0]["group"].split("_")[0]
        for index in group_range:
            new_group_id = f"{date}_{index}"
            for scene in group_info[date][new_group_id]:
                scenes.append(group_info[date][new_group_id][scene])
    else:
        if timeline_type == "after":
            max_group = max(group_ids)
            group_range = range(max_group + 1, max_group + TIMELINE_SPAN + 1)
        elif timeline_type == "before":
            min_group = min(group_ids)
            group_range = range(max(min_group - TIMELINE_SPAN, 0), min_group)
        else:
            raise NotImplementedError

        date = images[0]["group"].split("_")[0]
        for index in group_range:
            new_group_id = f"{date}_{index}"
            if new_group_id in group_info[date]:
                scenes.append([img for scene in group_info[date][new_group_id]
                                for img in group_info[date][new_group_id][scene]])

    return scenes

def get_full_scene_from_image(image, group_factor='group'):
    assert group_factor in ["group", "scene"], f"Invalid value of group_factor({group_factor}). Use \"scene\" or \"group\"."
    group_id = grouped_info_dict[image]["group"]
    date = group_id.split("_")[0]
    if group_factor == "group":
        return [img for scene in group_info[date][group_id]
                                for img in group_info[date][group_id][scene]]
    else:
        scene_id = grouped_info_dict[image]["scene"]
        return [img for img in group_info[date][group_id][scene_id]]

def get_multiple_scenes_from_images(begin_image, end_image, group_factor='group'):
    if group_factor == "group":
        min_group = grouped_info_dict[begin_image]["group"]
        max_group = grouped_info_dict[end_image]["group"]
        date = min_group.split("_")[0]
        min_group = int(min_group.split("_")[1])
        max_group = int(max_group.split("_")[1])
        group_range = range(min_group, max_group + 1)
        images = []
        for index in group_range:
            new_group_id = f"{date}_{index}"
            if new_group_id in group_info[date]:
                images.extend([img for scene in group_info[date][new_group_id]
                                for img in group_info[date][new_group_id][scene]])
    else:
        min_scene = grouped_info_dict[begin_image]["scene"]
        max_scene = grouped_info_dict[end_image]["scene"]
        date = min_scene.split("_")[0]
        min_scene = int(min_scene.split("_")[1])
        max_scene = int(max_scene.split("_")[1])
        scene_range = range(min_scene, max_scene + 1)
        images = []
        for index in scene_range:
            new_scene_id = f"{date}_{index}"
            if new_scene_id in scene_info:
                images.extend(scene_info[new_scene_id]["images"])
    return images


# For ARRON, use something like this
print(get_multiple_scenes_from_images("2016-08-15/20160815_052416_000.jpg", "2016-08-15/20160815_060939_000.jpg", group_factor="group"))
