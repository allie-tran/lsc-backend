from typing import List
import torch
from configs import FORCE_CPU, IMAGE_DIRECTORY
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from results.models import Event

processor_path = "Qwen/Qwen2-VL-2B-Instruct"
model_path = "lightonai/MonoQwen2-VL-v0.1"
device = "cpu"
if not FORCE_CPU and torch.cuda.is_available():
    device = "cuda"


class Reranker:
    def __init__(self, processor_path, model_path):
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    def score(self, query, image_paths):
        images = []
        for image_path in image_paths:
            image = Image.open(f"{IMAGE_DIRECTORY}/{image_path}")
            images.append(image)
        prompt = (
            "Assert the relevance of the previous image document to the following query/question, "
            "answer True or False. The query is: {query}"
        ).format(query=query)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Apply chat template and tokenize
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=text, images=images, return_tensors="pt").to("cuda")

        # Run inference to obtain logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_for_last_token = outputs.logits[:, -1, :]

        # Convert tokens and calculate relevance score
        true_token_id = self.processor.tokenizer.convert_tokens_to_ids("True")
        false_token_id = self.processor.tokenizer.convert_tokens_to_ids("False")
        relevance_score = torch.softmax(
            logits_for_last_token[:, [true_token_id, false_token_id]], dim=-1
        )

        # Extract and display probabilities
        true_prob = relevance_score[0, 0].item()
        return true_prob

    def rerank(self, query, image_paths):
        print(f"Reranking {len(image_paths)} images")
        scores = [self.score(query, [image_path]) for image_path in tqdm(image_paths)]
        return scores

    def rerank_scenes(self, query: str, scenes: List[Event]):
        scores = []
        print(f"Reranking {len(scenes)} scenes")
        for scene in tqdm(scenes):
            score = self.score(query, [image.src for image in scene.images][:1])
            scores.append(score)
        return scores


reranker = None
# reranker = Reranker(processor_path, model_path)
