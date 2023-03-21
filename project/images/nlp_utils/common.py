import re
import json
import shelve
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

FILE_DIRECTORY = f"/home/tlduyen/Deakin/processing/files/"
basic_dict = json.load(open(f"{FILE_DIRECTORY}/info_dict.json"))
all_people = json.load(open(f"{FILE_DIRECTORY}/all_people.json"))
all_dates = json.load(open(f"{FILE_DIRECTORY}/all_dates.json"))
all_visits = json.load(open(f"{FILE_DIRECTORY}/all_visits.json"))

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
