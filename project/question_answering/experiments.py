import auto_gptq
import torch
import torchvision
from auto_gptq.modeling._base import BaseGPTQForCausalLM
from PIL import Image
from transformers import AutoTokenizer
from transformers.image_processing_utils import Image

auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
torch.set_grad_enabled(False)


class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "model.layers"
    outside_layer_modules = [
        "vit",
        "vision_proj",
        "model.tok_embeddings",
        "model.norm",
        "output",
    ]
    inside_layer_modules = [
        ["attention.wqkv.linear"],
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]


model_name = "internlm/internlm-xcomposer2-vl-7b-4bit"

# init model and tokenizer
model = InternLMXComposer2QForCausalLM.from_quantized(
    model_name, trust_remote_code=True, device="cuda:0"
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model.tokenizer = tokenizer

# |%%--%%| <7YCgvEMzfY|vXHQPtNc8G>

def process_image(image):
    image = Image.open(image).convert("RGB")
    image = __resize_img__(image)
    image = model.vis_processor(image).cuda()
    return image

def encode_images(images):
    images = [process_image(image) for image in images]
    images = torch.stack(images).mean(0).unsqueeze(0)
    return images

def __resize_img__(b):
    width, height = b.size
    tar = max(width, height)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = int((tar - width) / 2)
    right_padding = tar - width - left_padding
    b = torchvision.transforms.functional.pad(
        b, [left_padding, top_padding, right_padding, bottom_padding]
    )
    return b


text = "<ImageHere>What kind of dog is this?"
images = ["201912/27/20191227_095008_000.jpg", "201909/08/20190908_071011_000.jpg"]
image_paths = [
    f"/home/allie/highres/LSC23/LSC23_highres_images/{image}" for image in images
]

with torch.cuda.amp.autocast():
    response, _ = model.chat(
        tokenizer, query=text, image=encode_images(image_paths), history=[], do_sample=False
    )
    print(response)


