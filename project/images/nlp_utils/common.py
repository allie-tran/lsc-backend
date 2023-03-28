import json
import re
import nltk
from nltk.corpus import stopwords
import os
import shelve

FILES_DIRECTORY = os.getenv("FILES_DIRECTORY")

stop_words = stopwords.words('english')
basic_dict = json.load(open(f"{FILES_DIRECTORY}/backend/basic_dict.json"))
locations = json.load(open(f'{FILES_DIRECTORY}/backend//locations.json'))
map_visualisation = json.load(open(f'{FILES_DIRECTORY}/backend/map_visualisation.json'))
regions = json.load(open(f'{FILES_DIRECTORY}/backend/regions.json'))
countries = json.load(open(f'{FILES_DIRECTORY}/backend/countries.json'))
lowercase_countries = {country.lower(): country for country in countries}

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
