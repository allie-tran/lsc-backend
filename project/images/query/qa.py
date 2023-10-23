import argparse
import transformers
from ..FrozenBiLM.args import get_args_parser
from ..FrozenBiLM.model import build_model, get_tokenizer
from ..FrozenBiLM.util.misc import get_mask, mask_tokens, adjust_learning_rate
from collections import defaultdict
from transformers import pipeline
import json
import torch 
from .common_nn import *
from transformers import pipeline
from .utils import basic_dict, datetime, scene_segments

transformers.logging.set_verbosity_error()
CLIP_EMBEDDINGS = "/mnt/DATA/duyen/highres"
qa_photo_features = np.load(f"{CLIP_EMBEDDINGS}/LSC23/ViT-L-14_openai_nonorm/features.npy")
qa_photo_ids = pd.read_csv(f"{CLIP_EMBEDDINGS}/LSC23/ViT-L-14_openai_nonorm/photo_ids.csv")["photo_id"].to_list()
DIM = qa_photo_features[0].shape[-1]
qa_photo_features = np.concatenate([qa_photo_features, 
                                    np.load(f"{CLIP_EMBEDDINGS}/LSC20/ViT-L-14_openai_nonorm/features.npy")])
qa_photo_ids += pd.read_csv(f"{CLIP_EMBEDDINGS}/LSC20/ViT-L-14_openai_nonorm/photo_ids.csv")["photo_id"].to_list()
image_to_id = {image: i for i, image in enumerate(photo_ids)}

options = ["frozenbilm_activitynet",
           "frozenbilm_tgif",
           "frozenbilm_msrvtt",
           "frozenbilm_msvd",
           "frozenbilm_tvqa"]

# text_qa_model_name = f"{PRETRAINED_MODELS}/models/question-answering__deepset--roberta-base-squad2"
text_qa_model_name = f"{PRETRAINED_MODELS}/models/bert-large-uncased-whole-word-masking-finetuned-squad"
# a) Get predictions
nlp = None
# device = "cpu"
# model_path = "/home/tlduyen/LQA/LLQA/RQ2/FrozenBiLM/lifelog_models/ftlifelogs_sub_all/ckpt.pth"
model_path = f"{PRETRAINED_MODELS}/models/FrozenBiLM/{options[0]}.pth"
parser = argparse.ArgumentParser(parents=[get_args_parser()])
args = parser.parse_args(f"""--combine_datasets msrvtt --combine_datasets_val msrvtt \
--suffix="." --max_tokens=256 --ds_factor_ff=8 --ds_factor_attn=8 \
--load={model_path} \
--msrvtt_vocab_path={PRETRAINED_MODELS}/datasets/MSRVTT-QA/vocab.json \
--model_name microsoft/deberta-v2-xlarge""".split())
if args.save_dir:
    args.save_dir = os.path.join(args.presave_dir, args.save_dir)

frozen_bilm = None
tokenizer = None
id2a = {}
def build_qa_model():
    global frozen_bilm
    global tokenizer
    global id2a
    global nlp
    # nlp = pipeline('question-answering', 
    #                model=text_qa_model_name, 
    #                tokenizer=text_qa_model_name, 
    #                topk=5, device=-1 if device=="cpu" else 0, truncation=True, padding=True)
    nlp = pipeline('question-answering', 
                   model=text_qa_model_name, 
                   tokenizer=text_qa_model_name, 
                   topk=5, device=-1, truncation=True, padding=True)
    # Build model
    print("Building QA model")
    tokenizer = get_tokenizer(args)
    # vocab = json.load(open(args.msrvtt_vocab_path, "r"))
    vocab = pd.read_csv(f"{FILES_DIRECTORY}/backend/all_answers.csv")["0"].to_list()[:3000]
    vocab = [a for a in vocab if a != np.nan and str(a) != "nan"]
    vocab = {a: i for i, a in enumerate(vocab)}
    id2a = {y: x for x, y in vocab.items()}
    args.n_ans = 2
    frozen_bilm = build_model(args)
    frozen_bilm.to(device)
    frozen_bilm.eval()
    n_parameters = sum(p.numel() for p in frozen_bilm.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # Load pretrained checkpoint
    assert args.load
    print("loading from", args.load)
    checkpoint = torch.load(args.load, map_location="cpu")
    frozen_bilm.load_state_dict(checkpoint["model"], strict=False)

    # Init answer embedding module
    aid2tokid = torch.zeros(len(vocab), args.max_atokens).long()
    for a, aid in vocab.items():
        try:
            tok = torch.tensor(
                tokenizer(
                    a,
                    add_special_tokens=False,
                    max_length=args.max_atokens,
                    truncation=True,
                    padding="max_length",
                )["input_ids"],
                dtype=torch.long,
            )
            aid2tokid[aid] = tok
        except ValueError as e:
            print(a, aid)
            raise(e)
    frozen_bilm.set_answer_embeddings(aid2tokid.to(device), freeze_last=args.freeze_last)

build_qa_model()

def answer(images, encoded_question):
    """
    Given a list of image filenames and an encoded question, generates the top 5 answers
    to the question based on the provided image embeddings.

    Args:
    - images: a list of strings representing image filenames
    - encoded_question: a dictionary containing the encoded question, including input_ids
      and attention_mask

    Returns:
    - A dictionary with the top 5 answers to the question and their corresponding confidence scores.
    """

    # Get image embeddings for the provided images
    image_embeddings = qa_photo_features[np.array(
        [image_to_id[image] for image in images if image])]
    
    # Convert image embeddings to PyTorch tensor and move to appropriate device
    video = torch.tensor(image_embeddings).to(device).float()

    # Subsample or pad the tensor if its length exceeds max_feats
    if len(video) >= args.max_feats:
        sampled = []
        for j in range(args.max_feats):
            sampled.append(video[(j * len(video)) // args.max_feats])
        video = torch.stack(sampled)
        video_len = args.max_feats
    else:
        video_len = len(video)
        video = torch.cat(
            [video, torch.zeros(args.max_feats - video_len, DIM).to(device)], 0
        )
    # Add an additional dimension to the tensor and move to appropriate device
    video = video.unsqueeze(0).to(device)
    
    # Create a mask for the tensor
    video_mask = get_mask(
        torch.tensor(video_len, dtype=torch.long).unsqueeze(0), video.size(1)
    ).to(device)

    # Move encoded question to appropriate device
    input_ids = encoded_question["input_ids"].to(device)
    attention_mask = encoded_question["attention_mask"].to(device)

    # Remove separator token and replace with padding token if not using suffix
    if not args.suffix:
        attention_mask[input_ids == tokenizer.sep_token_id] = 0
        input_ids[input_ids == tokenizer.sep_token_id] = tokenizer.pad_token_id

    # Use the provided BiLM to generate a prediction for the mask token
    output = frozen_bilm(
        video=video,
        video_mask=video_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Extract the logits for the mask token
    logits = output["logits"]
    delay = args.max_feats if args.use_video else 0
    logits = logits[:, delay: encoded_question["input_ids"].size(1) + delay][
        encoded_question["input_ids"] == tokenizer.mask_token_id
    ]  # get the prediction on the mask token
    logits = logits.softmax(-1)
    try:
        topk = torch.topk(logits, 5, -1)
        topk_txt = [[id2a[x.item()] for x in y] for y in topk.indices.cpu()]
        topk_scores = [[f"{x:.2f}".format() for x in y] for y in topk.values.cpu()]
        topk_all = [
            {x: y for x, y in zip(a, b)} for a, b in zip(topk_txt, topk_scores)
        ]
    except IndexError:
        topk_all = [{}]
    return topk_all[0]


def encode_question(question, textual_description=""):
    """
    Encodes a natural language question as a tokenized input suitable for input
    to a transformer model. The encoding includes special tokens to mark the beginning
    and end of the input, as well as a mask token to indicate where the answer
    should be predicted.
    
    Args:
    - question (str): a natural language question
    
    Returns:
    - encoded (torch.Tensor): a tensor of token ids representing the encoded question
    """
    # Capitalize and strip whitespace from the question string
    question = question.capitalize().strip()

    # If the question contains a [MASK] token, replace it with the mask token
    if "[MASK]" in question:
        question = question.replace("[MASK]", tokenizer.mask_token)
        text = f"{args.prefix} {question}{args.suffix}"
    # Otherwise, add "Question: " and "Answer: " tags to the question and mask the answer
    else:
        if question[-1] != "?":
            question = str(question) + "?"
        text = f"{args.prefix} Question: {question} Answer: {tokenizer.mask_token}. Subtitles: {textual_description} {args.suffix}"

    # Tokenize the text and encode the resulting token ids as a PyTorch tensor
    encoded = tokenizer(
        [text],
        add_special_tokens=True,
        max_length=args.max_tokens,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )

    return encoded

# Get textual description for a scene
def get_textual_description(scene, question):
    if nlp is None:
        build_qa_model()
    # converting datetime to string
    start_time = scene['start_time'].strftime("%I:%M%p")
    end_time = scene['end_time'].strftime("%I:%M%p")
    
    # calculating duration
    time_delta = scene['end_time'] - scene['start_time']
    if time_delta.seconds > 0:
        hours = time_delta.seconds // 3600
        if time_delta.days > 0:
            if hours > 0:
                duration = f"{time_delta.days} days and {hours} hours"
            else:
                duration = f"{time_delta.days} days"
        else:
            if hours > 0:
                minutes = (time_delta.seconds - hours * 3600) // 60
                duration = f"{hours} hours and {minutes} minutes"
            elif time_delta.seconds < 60:
                duration = f"{time_delta.seconds} seconds"
            else:
                minutes = time_delta.seconds // 60
                duration = f"{minutes} minutes"
        duration = f", lasted for about {duration}"
        time = f"from {start_time} to {end_time}"
    else:
        duration = ""
        time = f"at {start_time}"
        
    # converting datetime to string like Jan 1, 2020
    date = scene['start_time'].strftime("%A, %B %d, %Y")
    
    location = ""
    location_info = ""
    if scene["original_location"] != "---":
        location = f"in {scene['original_location']}"
        if scene["location_info"]:
            location_info = f", which is a {scene['location_info']}"
    ocr = ""
    if scene['ocr']:
        ocr = f"Some texts that can be seen from the images are: {' '.join(scene['ocr'])}."
    regions = [reg for reg in scene['region'] if reg != scene['country']]
    textual_description = f"The event happened {time}{duration} on {date} " + \
        f"{location}{location_info} in {', '.join(regions)} in {scene['country']}. {ocr}"
    return textual_description

# Get answers from a list of images:
def get_answers_from_images(images, question):
    answers = []
    # Filter out empty string images
    images = [image for image in images if image]
    print(images[0])
    
    # Build a scene from list of images
    scene = {}
    scene["start_time"] = basic_dict[images[0]]['time']
    scene["end_time"] = basic_dict[images[-1]]['time']
    scene["start_time"] = datetime.strptime(scene["start_time"], "%Y/%m/%d %H:%M:%S%z")
    scene["end_time"] = datetime.strptime(scene["end_time"], "%Y/%m/%d %H:%M:%S%z")
    
    first_image = basic_dict[images[0]]
    scene["original_location"] = first_image["location"]
    scene["location_info"] = first_image["location_info"]
    scene["country"] = first_image["country"]
    scene["region"] = first_image["region"]
    
    scene["ocr"] = []
    for image in images:
        for text in basic_dict[image]["ocr"]:
            if text and text not in scene["ocr"]:
                scene["ocr"].append(text)
                
    # Get textual description for a scene
    textual_description = get_textual_description(scene, question)
    # Get answers from textual description
    QA_input = {
                'question': question,
                'context': textual_description
            }
    print(textual_description)
    results = nlp(QA_input)
    print(results)
    for res in results:
        if res["score"] > 0.1:
            print(res["answer"], res["score"])
            answers.append(res["answer"])
    
    # Get answers from images
    encoded_question = encode_question(question)
    answers.extend(list(answer(images, encoded_question).keys()))
    
    return answers

def answer_topk_scenes(question, scenes, scores, k=10):
    """
    Given a natural language question and a list of scenes, returns the top k answers
    to the question across all scenes. Uses an encoding of the question and an answer
    function to compute answer scores for each scene.
    
    Args:
    - question (str): a natural language question
    - scenes (list of dicts): a list of scenes, each represented as a dictionary with
      the following keys:
        - "current" (list of tuples): a list of (image, score) with for the scene
    - k (int): the number of top scenes to consider
    
    Returns:
    - answers (list of str): the top 10 answers to the question across top-k scenes
    """
    # Create a defaultdict to accumulate answer scores across all scenes
    answers = defaultdict(float)
    if frozen_bilm is None:
        build_qa_model()
    # Encode the question using a helper function
    original_question = question
    
    # Iterate over each scene
    for i, (scene, score) in enumerate(zip(scenes[:k], scores[:k])):
        # Extract the images from the "current" field of the scene
        images = [i[0] for i in scene["current"] if i[0]]
        
        # Extract textual information from the scene
        scene["ocr"] = []
        textual_description = ""
        for image in images:
            for text in basic_dict[image]["ocr"]:
                if text and text not in scene["ocr"]:
                    scene["ocr"].append(text)
        try:
            textual_description = get_textual_description(scene, question)
            QA_input = {
                'question': original_question,
                'context': textual_description
            }
            results = nlp(QA_input)
            for res in results:
                if res["score"] > 0.1:
                    ans_score = res["score"] * (10-i) ** 2 / 5
                    answers[res["answer"]] = max(answers[res["answer"]], ans_score)
        except Exception as e:
            print("NLP", e)
            pass  
            
        encoded_question = encode_question(question, textual_description)

        # Compute answer scores for the current scene using an answer function
        ans = answer(images, encoded_question)
        
        # Accumulate the answer scores in the defaultdict
        for a, s in ans.items():
            answers[a] += float(s)
    
    # Sort the answers by score and take the top 10, discarding one if zero scores
    answers = [a for a, s in sorted(answers.items(), key=lambda x: x[1], reverse=True) if s > 0][:10]
    
    return answers

# question = "What is the color of the car in the cage?"
# answer(random.choices(range(len(photo_ids)), k=10), encode_question(question))