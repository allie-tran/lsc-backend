# import json
# from concurrent.futures import ThreadPoolExecutor
# from typing import Dict, List

# import auto_gptq
# import torch
# import torchvision
# from auto_gptq.modeling._base import BaseGPTQForCausalLM
# from configs import BUILD_ON_STARTUP
# from fsspec.asyn import asyncio
# from PIL import Image
# from transformers import AutoTokenizer
# from transformers.image_processing_utils import Image

# from llm.prompts import VISUAL_PROMPT

# auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
# torch.set_grad_enabled(False)


# class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
#     layers_block_name = "model.layers"
#     outside_layer_modules = [
#         "vit",
#         "vision_proj",
#         "model.tok_embeddings",
#         "model.norm",
#         "output",
#     ]
#     inside_layer_modules = [
#         ["attention.wqkv.linear"],
#         ["attention.wo.linear"],
#         ["feed_forward.w1.linear", "feed_forward.w3.linear"],
#         ["feed_forward.w2.linear"],
#     ]


# class InternLMInference:
#     def __init__(self):
#         self.loaded = False

#     def load_model(self):
#         model_name = "internlm/internlm-xcomposer2-vl-7b-4bit"

#         # init model and tokenizer
#         self.model = InternLMXComposer2QForCausalLM.from_quantized(
#             model_name, trust_remote_code=True, device="cuda:0"
#         ).eval()

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_name, trust_remote_code=True
#         )
#         self.model.tokenizer = self.tokenizer
#         self.loaded = True

#     @staticmethod
#     def __resize_img__(b):
#         width, height = b.size
#         tar = max(width, height)
#         top_padding = int((tar - height) / 2)
#         bottom_padding = tar - height - top_padding
#         left_padding = int((tar - width) / 2)
#         right_padding = tar - width - left_padding
#         b = torchvision.transforms.functional.pad(  # type: ignore
#             b, (left_padding, top_padding, right_padding, bottom_padding)
#         )
#         b = b.resize((224, 224))
#         return b

#     async def process_image(self, image) -> torch.Tensor:
#         loop = asyncio.get_running_loop()
#         with ThreadPoolExecutor() as pool:
#             image = await loop.run_in_executor(pool, Image.open, image)
#             image = await loop.run_in_executor(pool, image.convert, "RGB")
#             image = await loop.run_in_executor(pool, self.__resize_img__, image)
#             image = self.model.vis_processor(image).cuda()

#         # image = Image.open(image).convert("RGB")
#         # image = self.__resize_img__(image)
#         # image = self.model.vis_processor(image).cuda()
#         return image

#     async def encode_images(self, images: List[str]) -> torch.Tensor:
#         """
#         What we can do is that we can encode the images first and then pass them to the model
#         """
#         processed = await asyncio.gather(
#             *[self.process_image(image) for image in images]
#         )
#         # processed = [self.process_image(image) for image in images]
#         embed_images = torch.stack(processed).mean(0).unsqueeze(0)
#         return embed_images

#     async def answer_question(
#         self, question: str, images: List[str], extra_info: str
#     ) -> Dict[str, str]:
#         embed_images = await self.encode_images(images)
#         # embed_images = self.encode_images(images)
#         text= VISUAL_PROMPT.format(
#             question=question,
#             extra_info=extra_info,
#         )
#         with torch.cuda.amp.autocast():
#             response, _ = self.model.chat(
#                 self.tokenizer,
#                 query=text,
#                 image=embed_images,
#                 history=[]
#             )
#         try:
#             response = json.loads(response)
#             return response
#         except json.JSONDecodeError:
#             print(response)
#             return {}


# internlm_model = InternLMInference()
# if BUILD_ON_STARTUP:
#     internlm_model.load_model()
internlm_model = None
