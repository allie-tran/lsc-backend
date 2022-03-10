import json
import os
from .utils import *
from ..nlp_utils.common import FILE_DIRECTORY, COMMON_DIRECTORY, basic_dict

TIMELINE_SPAN = 9  # If they want more, submit more
groups = json.load(open(f"{COMMON_DIRECTORY}/group_segments.json"))
scene_segments = {}
for group_name in groups:
    for scene_name, images in groups[group_name]["scenes"]:
        assert "S_" in scene_name, f"{scene_name} is not a valid scene id"
        scene_segments[scene_name] = images
time_info = json.load(open(f"{FILE_DIRECTORY}/time_info.json"))

def get_next_scenes(date, group_range, max_scene):
    scenes = []
    for index in group_range:
        new_group_id = f"{date}_{index}"
        if new_group_id in group_segments[date]:
            for scene in group_segments[date][new_group_id]:
                if int(scene.split('_')[-1]) >= max_scene:
                    scenes.append(group_segments[date][new_group_id][scene])
                    if len(scenes) > TIMELINE_SPAN:
                        return scenes, -1
    return scenes, -1


def get_prev_scenes(date, group_range, min_scene):
    scenes = []
    group_range = list(group_range)[::-1]
    for index in group_range:
        new_group_id = f"{date}_{index}"
        if new_group_id in group_segments[date]:
            for scene in group_segments[date][new_group_id]:
                if int(scene.split('_')[-1]) <= min_scene:
                    scenes.append(group_segments[date][new_group_id][scene])
                    if len(scenes) > TIMELINE_SPAN:
                        return scenes, -1
    return scenes, -1

def get_timeline(images, direction="full"):
    images = [basic_dict[image]for image in images]
    scene_id = int(images[0]["scene"].split('_S')[-1])
    group_id = int(images[0]["group"].split('_G')[-1])
    scenes = []
    date = images[0]["scene"].split("_")[0]
    marked = -1

    if direction == "full":
        scene_range = range(scene_id-10, scene_id + 10)
    elif direction == "next":
        scene_range = range(scene_id, scene_id + 10)
    elif direction == "previous":
        scene_range = range(scene_id - 10, scene_id)
    for index in scene_range:
        scene_name = f"{date}_S{index}"
        if scene_name in scene_segments[date]:
            if index == scene_id:
                marked = len(scenes)
            scenes.append((scene_segments[date][scene_name], time_info[scene_name]))

    return scenes, marked, group_id

def get_timeline_group(date):
    groups = []
    for group in group_segments[date]:
        groups.append((list(group_segments[date][group].values())[0][0], time_info[group]))
    return groups


def get_full_scene_from_image(image, group_factor='group'):
    assert group_factor in [
        "group", "scene"], f"Invalid value of group_factor({group_factor}). Use \"scene\" or \"group\"."
    group_id = basic_dict[image]["group"]
    date = group_id.split("_")[0]
    if group_factor == "group":
        return [img for scene in group_segments[date][group_id]
                for img in group_segments[date][group_id][scene]]
    else:
        scene_id = basic_dict[image]["scene"]
        return [img for img in group_segments[date][group_id][scene_id]]


def get_multiple_scenes_from_images(begin_image, end_image, group_factor='group'):
    min_group = basic_dict[begin_image]["group"]
    max_group = basic_dict[end_image]["group"]
    date = min_group.split("_")[0]
    min_group = int(min_group.split("_")[1])
    max_group = int(max_group.split("_")[1])
    group_range = range(min_group, max_group + 1)

    if group_factor == "group":
        images = []
        for index in group_range:
            new_group_id = f"{date}_{index}"
            if new_group_id in group_segments[date]:
                images.extend([img for scene in group_segments[date][new_group_id]
                               for img in group_segments[date][new_group_id][scene]])
    else:
        min_scene = basic_dict[begin_image]["scene"]
        max_scene = basic_dict[end_image]["scene"]
        min_scene = int(min_scene.split("_")[1])
        max_scene = int(max_scene.split("_")[1])
        images = []
        scene_range = range(min_scene, max_scene + 1)
        for index in group_range:
            new_group_id = f"{date}_{index}"
            if new_group_id in group_segments[date]:
                images.extend([img for scene in group_segments[date][new_group_id]
                               for img in group_segments[date][new_group_id][scene] if int(scene.split('_')[1]) in scene_range])
    return images


def to_full_key(image):
    return f"{image[:6]}/{image[6:8]}/{image}"

# NEW LSC22
def get_all_scenes(images):
    images = [basic_dict[image]for image in images]
    scene_id = images[0]["scene"]
    group_id = int(images[0]["group"].split('G_')[-1])
    date = images[0]["scene"].split("_")[0]
    group_results = []
    group_range = range(group_id - 1, group_id + 2)
    group_range = [f"G_{index}" for index in group_range]
    print(group_range)
    line = 0
    done = False
    space = 0
    for group in group_range:
        if group in groups:
            scenes = []
            for scene_name, images in groups[group]["scenes"]:
                scenes.append(
                    (scene_name, images, time_info[scene_name]))
                if scene_id == scene_name:
                    line += (len(scenes) - 1) // 4 + 1
                    done = True
            if scenes:
                if not done:
                    space += 1
                    line += (len(scenes) - 1) // 4 + 1
                group_results.append(
                    (group, groups[group]["location"], scenes))
    return group_results, line, space, scene_id
