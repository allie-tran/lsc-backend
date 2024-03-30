from query_parse.constants import BASIC_DICT

from .utils import *

TIMELINE_SPAN = 9  # If they want more, submit more


def get_submission(image, scene):
    submissions = []
    if scene:
        images = scene_segments[BASIC_DICT[image]["scene"]]
        for img in images:
            submissions.append(img.split(".")[0].split("/")[-1])
    else:
        submissions.append(image.split(".")[0].split("/")[-1])
    return submissions


def get_image_list(first, last):
    first = all_images.index(first)
    last = all_images.index(last)
    return [img.split(".")[0].split("/")[-1] for img in all_images[first : last + 1]]


# NEW LSC22
def get_all_scenes(images):
    print(images)
    images = [BASIC_DICT[image] for image in images if image]
    scene_id = images[0]["scene"]
    group_id = int(images[0]["group"].split("G_")[-1])
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
                scenes.append((scene_name, images, time_info[scene_name]))
                if scene_id == scene_name:
                    line += (len(scenes) - 1) // 4 + 1
                    done = True
            if scenes:
                if not done:
                    space += 1
                    line += (len(scenes) - 1) // 4 + 1
                group_results.append(
                    (
                        group,
                        [
                            groups[group]["location"],
                            str(groups[group]["location_info"]),
                        ],
                        scenes,
                    )
                )

    print("Line:", line, ", scene_id", scene_id)
    return group_results, line, space, scene_id


def get_more_scenes(group_id, direction="top"):
    group_id = int(group_id.split("G_")[-1])
    group_results = []
    if direction == "bottom":
        group_range = range(group_id + 1, group_id + 3)
    else:
        group_range = range(group_id - 2, group_id)
    line = 0
    space = 0
    group_range = [f"G_{index}" for index in group_range]
    for group in group_range:
        if group in groups:
            scenes = []
            for scene_name, images in groups[group]["scenes"]:
                scenes.append((scene_name, images, time_info[scene_name]))
            if scenes:
                space += 1
                line += (len(scenes) - 1) // 4 + 1
                group_results.append(
                    (
                        group,
                        [
                            groups[group]["location"],
                            str(groups[group]["location_info"]),
                        ],
                        scenes,
                    )
                )
    return group_results, line, space
