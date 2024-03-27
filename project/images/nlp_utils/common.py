import json
import os
import re
import shelve
from typing import Generator, List, Union

import nltk
from nltk.corpus import stopwords

from ..nlp_utils.types import RegexInterval

FILES_DIRECTORY = os.getenv("FILES_DIRECTORY")

STOP_WORDS = stopwords.words("english")
BASIC_DICT = json.load(open(f"{FILES_DIRECTORY}/backend/basic_dict.json"))
LOCATIONS = json.load(open(f"{FILES_DIRECTORY}/backend//locations.json"))
LOCATION_INFOS = json.load(open(f"{FILES_DIRECTORY}/backend//location_info.json"))
MAP_VISUALISATION = json.load(open(f"{FILES_DIRECTORY}/backend/map_visualisation.json"))
REGIONS = json.load(open(f"{FILES_DIRECTORY}/backend/regions.json"))
REGIONS.extend(["korea", "uk", "england"])
COUNTRIES = json.load(open(f"{FILES_DIRECTORY}/backend/countries.json"))
LOWERCASE_COUNTRIES = {country.lower(): country for country in COUNTRIES}

GPS_NORMAL_CASE = {}
for loc in LOCATIONS:
    for origin_doc, (lat, lon) in MAP_VISUALISATION:
        if loc == origin_doc.lower():
            GPS_NORMAL_CASE[loc] = origin_doc


def find_regex(regex, text, escape=False) -> Generator[RegexInterval, None, None]:
    if "\n" in regex or ("#" in regex and "\\#" not in regex):
        regex = re.compile(regex, re.IGNORECASE | re.VERBOSE)
    else:
        regex = re.compile(regex, re.IGNORECASE)
    for m in regex.finditer(text):
        result = m.group()
        start = m.start()
        while len(result) > 0 and result[0] == " ":
            result = result[1:]
            start += 1
        while len(result) > 0 and result[-1] == " ":
            result = result[:-1]
        yield RegexInterval(start=start, end=start + len(result), text=result)


def flatten_tree(t: Union[str, nltk.tree.Tree]) -> str:
    if isinstance(t, str):
        return t
    return " ".join([l[0] for l in t.leaves()])


def flatten_tree_tags(
    t: Union[str, nltk.tree.Tree], pos_str: List[str], pos_tree: List[str]
) -> Union[str, List]:
    if isinstance(t, nltk.tree.Tree):
        if t.label() in pos_str:
            return [flatten_tree(t), t.label()]
        elif t.label() in pos_tree:
            return [t, "tree"]
        else:
            return [flatten_tree_tags(l, pos_str, pos_tree) for l in t]
    else:
        return t


def cache(_func=None, *, file_name=None, separator="_"):
    """
    if file_name is None, just cache it using memory, else save result to file
    """
    if file_name:
        d = shelve.open(file_name)
    else:
        d = {}

    def decorator(func):
        def new_func(*args, **kwargs):
            param = separator.join(
                [str(arg) for arg in args] + [str(v) for v in kwargs.values()]
            )
            if param not in d:
                d[param] = func(*args, **kwargs)
            return d[param]

        return new_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)
