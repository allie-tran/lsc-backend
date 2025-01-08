import argparse
import os

import numpy as np
import pandas as pd
import torch
from configs import (
    BUILD_ON_STARTUP,
    MSRVTT_VOCAB_PATH,
    PRETRAINED_MODELS,
    QA_LLM_BASE,
    QA_PATH,
    TEXT_QA_MODEL,
    VOCAB_PATH,
)
from query_parse.visual import device
from transformers.pipelines import pipeline

from .FrozenBiLM.args import get_args_parser
from .FrozenBiLM.model import build_model, get_tokenizer

# Get FrozenBiLM arguments
parser = argparse.ArgumentParser(parents=[get_args_parser()])

args = parser.parse_args(
    f"""--combine_datasets msrvtt --combine_datasets_val msrvtt \
--suffix="." --max_tokens=256 --ds_factor_ff=8 --ds_factor_attn=8 \
--load={QA_PATH} \
--msrvtt_vocab_path={MSRVTT_VOCAB_PATH},
--model_name {QA_LLM_BASE}""".split()
)
if args.save_dir:
    args.save_dir = os.path.join(args.presave_dir, args.save_dir)


class FrozenBiLMInference:
    def __init__(self):
        self.loaded = False

    def get_vocab(self):
        """
        Load the vocabulary for the MSRVTT dataset and return a dictionary mapping
        words to their corresponding token ids.
        """
        vocab = pd.read_csv(VOCAB_PATH)["0"].to_list()[:3000]  # type: ignore
        vocab = [a for a in vocab if a != np.nan and str(a) != "nan"]
        vocab = {a: i for i, a in enumerate(vocab)}
        id2a = {y: x for x, y in vocab.items()}

        self.vocab = vocab
        self.id2a = id2a

    def build_text_qa(self):
        """
        Build a text question-answering model using the provided arguments and load a pretrained
        checkpoint if available.
        """
        text_model_path = f"{PRETRAINED_MODELS}/{TEXT_QA_MODEL}"
        text_qa = pipeline(
            "question-answering",
            model=text_model_path,
            tokenizer=text_model_path,
            top_k=2,
            device=-1,
            truncation=True,
            padding=True,
        )

        self.text_qa = text_qa

    def build_visual_qa(self):
        # Build model
        print("Building QA model")
        tokenizer = get_tokenizer(args)
        args.n_ans = 2
        visual_qa = build_model(args)
        assert isinstance(visual_qa, torch.nn.Module), "Model is not a torch.nn.Module"
        visual_qa.to(device)
        visual_qa.eval()

        # Load pretrained checkpoint
        assert args.load
        print("Loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        visual_qa.load_state_dict(checkpoint["model"], strict=False)

        # Init answer embedding module
        self.get_vocab()
        vocab = self.vocab
        id2a = self.id2a

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
                raise (e)
        visual_qa.set_answer_embeddings(
            aid2tokid.to(device), freeze_last=args.freeze_last
        )
        self.visual_qa = visual_qa
        self.tokenizer = tokenizer

    def build(self):
        """
        Build the text and visual question-answering models.
        """
        self.build_text_qa()
        # self.build_visual_qa()
        self.loaded = True


qa_model = FrozenBiLMInference()
if BUILD_ON_STARTUP:
    qa_model.build()
