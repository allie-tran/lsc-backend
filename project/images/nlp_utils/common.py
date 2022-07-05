import json
import re

import nltk
from nltk.corpus import stopwords
import os
import shelve
import joblib
import numpy as np

stop_words = stopwords.words('english')
FILE_DIRECTORY = "/home/tlduyen/LSC22/process/files/backend"
COMMON_DIRECTORY = "/home/tlduyen/LSC22/process/files/"
basic_dict = json.load(open(f"{FILE_DIRECTORY}/basic_dict.json"))

locations = json.load(open(f'{FILE_DIRECTORY}/locations.json'))
map_visualisation = json.load(open(f'{FILE_DIRECTORY}/map_visualisation.json'))
# countries = ["England", "United Kingdom", "China", "Ireland", "Germany", "Greece", "Thailand", "Vietnam", "Spain", "Turkey", "Korea", "France", "Switzerland", "Australia", "Denmark", "Romania", "Norway"]
regions = json.load(open(f'{FILE_DIRECTORY}/regions.json'))
countries = json.load(open(f'{FILE_DIRECTORY}/countries.json'))


all_keywords = json.load(open(f'{FILE_DIRECTORY}/all_keywords.json'))

def find_regex(regex, text, escape=False):
    regex = re.compile(regex, re.IGNORECASE + re.VERBOSE)
    for m in regex.finditer(text):
        result = m.group()
        start = m.start()
        while len(result) > 0 and result[0] == ' ':
            result = result[1:]
            start += 1
        while len(result) > 0 and result[-1] == ' ':
            result = result[:-1]
        yield (start, start + len(result), result)


def flatten_tree(t):
    if isinstance(t, str):
        return t
    return " ".join([l[0] for l in t.leaves()])


def flatten_tree_tags(t, pos_str, pos_tree):
    if isinstance(t, nltk.tree.Tree):
        if t.label() in pos_str:
            return [flatten_tree(t), t.label()]
        elif t.label() in pos_tree:
            return [t, "tree"]
        else:
            return [flatten_tree_tags(l, pos_str, pos_tree) for l in t]
    else:
        return t


def cache(_func=None, *, file_name=None, separator='_'):
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
                [str(arg) for arg in args] + [str(v) for v in kwargs.values()])
            if param not in d:
                d[param] = func(*args, **kwargs)
            return d[param]
        return new_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)


overlap_keywords = json.load(open(f"{FILE_DIRECTORY}/overlap_keywords.json"))

@cache
def intersect(word, keyword):
    if word == keyword:
        return True
    try:
        if word in keyword.split(' '):
            cofreq = overlap_keywords[word][keyword]
            # return True
            return cofreq / freq[word] > 0.9
        elif keyword in word.split(' '):
            cofreq = overlap_keywords[keyword][word]
            # return True
            return cofreq / all_keywords[keyword] > 0.8
    except (KeyError, AttributeError) as e:
        pass
    return False
