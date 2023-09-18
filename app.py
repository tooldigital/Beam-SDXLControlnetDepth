from beam import App, Runtime, Image, Output, Volume

import os
import time
import torch
import shutil
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
import base64
import random
from io import BytesIO

cache_path = "./models"
MAX_SEED = np.iinfo(np.int32).max

app = App(
    name="sdxl",
    runtime=Runtime(
        cpu=8,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.10",
            python_packages=[
                "accelerate==0.17.0",
                "diffusers==0.19.0",
                "invisible-watermark==0.2.0",
                "Pillow==10.0.0",
                "torch==2.0.1",
                "transformers==4.27",
                "xformers==0.0.21" 
            ],
            commands=["apt-get update && apt-get install ffmpeg libsm6 libxext6  -y"],
        ),
    ),
    volumes=[Volume(name="models", path="./models")],
)


def load_models():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading depth feature extractor")
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    print("Loading controlnet depth model")

    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0",use_safetensors=True,torch_dtype=torch.float16)
    print("Loading better VAE")
    better_vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",torch_dtype=torch.float16)
    print("Loading sdxl")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=better_vae,
        controlnet=controlnet,
        use_safetensors=True,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)

    return pipe

def get_depth_map(image, feature_extractor, depth_estimator):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")

    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

@app.rest_api(loader=load_models, keep_warm_seconds=300)
def doControlNETSDXLDepth(**inputs):
    # Grab inputs passed to the API

    pipe = inputs["context"]

    prompt = inputs["prompt"]
    seed = inputs["seed"]
    w = inputs["width"]
    h = inputs["height"]


    if(seed == 0):
        seed = random.randint(0, MAX_SEED)
    
    generator = torch.Generator().manual_seed(seed)


   # buffered = BytesIO()
  #  image.save(buffered, format='JPEG',quality=80)
 #   img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
 #   return {"image": img_str}

    return {"image": "hello"}


if __name__ == "__main__":
    print("main called")
    # You can customize this query however you want:
    # urls = ["https://www.nutribullet.com"]
    # query = "What are some use cases I can use this product for?"
    # start_conversation(urls=urls, query=query)
