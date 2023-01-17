import json
import re

import nltk
from nltk.corpus import stopwords
import os
import shelve
import joblib
import numpy as np

stop_words = stopwords.words('english')
stop_words += [',', '.']

COMMON_PATH = os.getenv("COMMON_PATH")
simpletime = ['at', 'around', 'about', 'on']
period = ['while', "along", "as"]

preceeding = ['before', "afore"]
following = ['after']
location = ['across', 'along', 'around', 'at', 'behind', 'beside', 'near', 'by', 'nearby', 'close to',
            'next to', 'from', 'in front of', 'inside', 'in', 'into', 'off', 'on',
            'opposite', 'out of', 'outside', 'past', 'through', 'to', 'towards']

all_words = period + preceeding + following
all_prep = simpletime + period + preceeding + following
pattern = re.compile(f"\s?({'|'.join(all_words)}+)\s")

# grouped_info_dict = json.load(open(f"{COMMON_PATH}/grouped_info_dict.json"))
# locations = set([img["location"].lower()
#                  for img in grouped_info_dict.values()])
# regions = set([w.strip().lower() for img in grouped_info_dict.values()
#                for w in img["region"]])
# deeplab = set([w.replace('_', ' ') for img in grouped_info_dict.values()
#                for w in img["deeplab"]])
# coco = set([w.replace('_', ' ') for img in grouped_info_dict.values()
#             for w in img["coco"]])
# attributes = set([w.replace('_', ' ') for img in grouped_info_dict.values()
#                   for w in img["attributes"]])
# category = set([w.replace('_', ' ') for img in grouped_info_dict.values()
#                 for w in img["category"]])
# microsoft = set([w.replace('_', ' ') for img in grouped_info_dict.values()
#                  for w in img["microsoft_tags"] + img["microsoft_descriptions"]])

locations = json.load(open(f'{COMMON_PATH}/locations.json'))
map_visualisation = json.load(open(f'{COMMON_PATH}/map_visualisation.json'))
countries = ["England", "United Kingdom", "China", "Ireland", "Czech Republic", "Germany", "Belarus", "Belgium", "Netherlands", "Norway", "Poland", "Russia", "Sweden", "Turkey"]
regions = json.load(open(f'{COMMON_PATH}/regions.json'))
microsoft = json.load(open(f'{COMMON_PATH}/microsoft.json'))
coco = json.load(open(f'{COMMON_PATH}/coco.json'))

all_keywords = json.load(open(f'{COMMON_PATH}/all_keywords.json'))
all_keywords_without_attributes = json.load(
    open(f'{COMMON_PATH}/all_keywords_without_attributes.json'))
# attribute_keywords = json.load(open(f'{COMMON_PATH}/attribute_keywords.json'))
attribute_keywords = ['beige',
                           'black',
                           'blonde',
                           'blue',
                           'blurry',
                           'brick',
                           'bright',
                           'brown',
                           'ceramic',
                           'checkered',
                           'circular',
                           'clear',
                           'closed',
                           'cloudy',
                           'colorful',
                           'cooked',
                           'curly',
                           'curved',
                           'dark',
                           'decorated',
                           'decorative',
                           'digital',
                           'dirty',
                           'dry',
                           'electrical',
                           'empty',
                           'flat',
                           'flat screen',
                           'floral',
                           'fluffy',
                           'framed',
                           'full',
                           'glass',
                           'glazed',
                           'gold',
                           'golden',
                           'granite',
                           'gray',
                           'green',
                           'hairy',
                           'khaki',
                           'large',
                           'leafless',
                           'leather',
                           'little',
                           'long',
                           'marble',
                           'maroon',
                           'metal',
                           'old',
                           'open',
                           'orange',
                           'paper',
                           'parked',
                           'patterned',
                           'paved',
                           'piled',
                           'pink',
                           'plaid',
                           'plastic',
                           'potted',
                           'purple',
                           'red',
                           'reflecting',
                           'ripe',
                           'rocky',
                           'rolled',
                           'round',
                           'rusty',
                           'sandy',
                           'sharp',
                           'shining',
                           'shiny',
                           'shirtless',
                           'short',
                           'silver',
                           'sleeping',
                           'sliced',
                           'small',
                           'smiling',
                           'snowy',
                           'square',
                           'stacked',
                           'stainless steel',
                           'stone',
                           'straw',
                           'striped',
                           'stuffed',
                           'tall',
                           'thick',
                           'thin',
                           'tiled',
                           'walking',
                           'watching',
                           'wet',
                           'white',
                           'wicker',
                           'wooden',
                           'yellow',
                           'young']

basic_dict = json.load(open(f"{COMMON_PATH}/basic_dict.json"))
all_address = '|'.join([re.escape(a) for a in locations])
activities = set(["walking", "airplane", "transport", "running"])
# phrases = json.load(open(f'{COMMON_PATH}/phrases.json'))
_, aIDF = joblib.load(f"{COMMON_PATH}/aTFIDF.joblib")
aIDF_indices = {keyword: i for (i, keyword) in enumerate(aIDF.keys())}
ocr_keywords = [word for word in json.load(open(f"{COMMON_PATH}/ocr_keywords.json")) if len(word) > 1]
_, attributeIDF = joblib.load(f"{COMMON_PATH}/attribute_tfidf.joblib")
_, ocrIDF = joblib.load(f"{COMMON_PATH}/ocr_tfidf.joblib")

map2deeplab = json.load(open(f"{COMMON_PATH}/map2deeplab.json"))
deeplab2simple = json.load(open(f"{COMMON_PATH}/deeplab2simple.json"))
simples = json.load(open(f"{COMMON_PATH}/simples.json"))
def to_deeplab(word):
    for kw in map2deeplab:
        if word in map2deeplab[kw][1]:
            yield deeplab2simple[kw]

def to_vector(scores):
    vector = [0 for _ in range(len(aIDF_indices))]
    for keyword, score in scores.items():
        for word in to_deeplab(keyword):
            vector[aIDF_indices[word]] += score
    return vector

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


freq = json.load(open(f"{COMMON_PATH}/stats/all.json"))
overlap = json.load(open(f"{COMMON_PATH}/stats/overlap.json"))


@cache
def intersect(word, keyword):
    if word == keyword:
        return True
    try:
        if word in keyword.split(' '):
            cofreq = overlap[word][keyword]
            # return True
            return cofreq / freq[word] > 0.9
        elif keyword in word.split(' '):
            cofreq = overlap[keyword][word]
            # return True
            return cofreq / freq[keyword] > 0.8
    except (KeyError, AttributeError) as e:
        pass
    return False
