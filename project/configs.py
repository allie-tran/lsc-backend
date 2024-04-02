"""
Configs for the project
"""

import os

# ====================== #
# General Customisation  #
# ====================== #
# These could be adjusted in the frontend with settings
DEV_MODE = True
FILTER_FIELDS = True
MERGE_EVENTS = True
MAX_IMAGES_PER_EVENT = 20
DEFAULT_SIZE = 200
TIMELINE_SPAN = 9  # If they want more, submit more

# ==================== #
# Model Configurations #
# ==================== #

# CLIP model configurations
FORCE_CPU = False
MODEL_NAME = "ViT-L-14-336"  # or ViT-H-14
PRETRAINED_DATASET = "openai"  # or laion2b_s32b_b79k
EMBEDDING_DIM = 768
CLIP_EMBEDDINGS = os.environ.get("CLIP_EMBEDDINGS")
# ==================== #

# ====================== #
# Dataset Configurations #
# ====================== #
DATA_YEARS = ["LSC20", "LSC23"]
FILES_DIRECTORY = os.getenv("FILES_DIRECTORY")
# ====================== #
# Elasticsearch Configurations #
# ====================== #
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = os.getenv("ES_PORT", 9200)
ES_URL = f"http://{ES_HOST}:{ES_PORT}"

INDEX = os.getenv("INDEX", "all_lsc")
SCENE_INDEX = os.getenv("SCENE_INDEX", "all_lsc_mean")
CLIP_MIN_SCORE = 1.2  # 1.2 is the normal score, 1.0 is for Transf

INCLUDE_SCENE = ["scene"]
INCLUDE_FULL_SCENE = [
    "images",
    "start_time",
    "end_time",
    "gps",
    "scene",
    "group",
    "timestamp",
    "location",
    "cluster_images",
    "weights",
    "country",
    "ocr",
    "country",
    "location_info",
    "duration",
    "region",
    "date",
]
INCLUDE_IMAGE = ["image_path", "time", "gps", "scene", "group", "location"]

# ========================== #
# Functions to derive fields #
# ========================== #
ESSENTIAL_FIELDS = ["images", "scene", "group", "start_time", "end_time"]
DERIVABLE_FIELDS = {
    "minute": lambda x: x.start_time.strftime("%H:%M"),
    "hour": lambda x: x.start_time.strftime("%H %p"),
    "date": lambda x: x.start_time.strftime("%d-%m-%Y"),
    "week": lambda x: x.start_time.isocalendar()[1],
    "month": lambda x: x.start_time.strftime("%B %Y"),
    "year": lambda x: x.start_time.year,
    "city": lambda x: [r for r in x.region if r != x.country],
    "days": lambda x: (x.end_time - x.start_time).days,
    "hours": lambda x: (x.end_time - x.start_time).seconds // 3600,
    "weeks": lambda x: (x.end_time - x.start_time).days // 7,
}

ISEQUAL = {"*": lambda x, y: x == y, "city": lambda x, y: set(x).intersection(set(y))}
# Fields in ORDER in the textual description
TIME_FIELDS = ["minute", "hour", "weekday", "date", "week", "month", "year"]
LOCATION_FIELDS = ["location", "location_info", "city", "region", "country"]
DURATION_FIELDS = ["months", "weeks", "days", "hours", "minutes"]
VISUAl_FIELDS = ["images", "ocr"]

# ====================== #
# QA Configurations #
# ====================== #

BUILD_ON_STARTUP = True
PRETRAINED_MODELS = os.environ.get("PRETRAINED_MODELS")
QA_FEATURES = [
    f"{CLIP_EMBEDDINGS}/LSC23/ViT-L-14-336_openai_nonorm/features.npy",
    f"{CLIP_EMBEDDINGS}/LSC20/ViT-L-14-336_openai_nonorm/features.npy",
]
QA_IDS = [
    f"{CLIP_EMBEDDINGS}/LSC23/ViT-L-14-336_openai_nonorm/photo_ids.csv",
    f"{CLIP_EMBEDDINGS}/LSC20/ViT-L-14-336_openai_nonorm/photo_ids.csv",
]
QA_DIM = 768

QA_OPTIONS = [
    "frozenbilm_activitynet",
    "frozenbilm_tgif",
    "frozenbilm_msrvtt",
    "frozenbilm_msvd",
    "frozenbilm_tvqa",
]

QA_PATH = f"{PRETRAINED_MODELS}/models/FrozenBiLM/{QA_OPTIONS[0]}.pth"
MSRVTT_VOCAB_PATH = f"{PRETRAINED_MODELS}/datasets/MSRVTT-QA/vocab.json"
VOCAB_PATH = f"{FILES_DIRECTORY}/backend/all_answers.csv"
QA_LLM_BASE = "microsoft/deberta-v2-xlarge"

TEXT_QA_MODEL_OPTIONS = [
    "models/question-answering__deepset--roberta-base-squad2",
    "models/bert-large-uncased-whole-word-masking-finetuned-squad",
]

TEXT_QA_MODEL = TEXT_QA_MODEL_OPTIONS[1]

JSON_START_FLAG = "```json"
JSON_END_FLAG = "```"
