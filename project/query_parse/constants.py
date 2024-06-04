import json

from configs import FILES_DIRECTORY
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words("english")
# BASIC_DICT = json.load(open(f"{FILES_DIRECTORY}/backend/basic_dict.json"))
BASIC_DICT = {}
DESCRIPTIONS = json.load(open(f"{FILES_DIRECTORY}/backend/tags.json"))
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

QUESTION_WORDS = [
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "whose",
    "whom",
]

QUESTION_TYPES = [
    "location",
    "time",
    "frequency",
    "ocr",
    "counting",
    "summarization",
    "color",
]

AUXILIARY_VERBS = [
    "do",
    "have",
    "can",
    "could",
    "shall",
    "should",
    "will",
    "would",
    "did",
    "does",
    "had",
    "was",
    "were",
    "is",
    "are",
    "am",
    "may",
]
