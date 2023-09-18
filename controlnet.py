import os
import time
import torch
import shutil
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image



CONTROL_MODEL = "diffusers/controlnet-depth-sdxl-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
FEATURE_NAME = "Intel/dpt-hybrid-midas"
CONTROL_CACHE = "control-cache"
VAE_CACHE = "vae-cache"
MODEL_CACHE = "sdxl-cache"
FEATURE_CACHE = "feature-cache"

t1 = time.time()
print("Loading depth feature extractor")
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
print("Loading controlnet depth model")

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    use_safetensors=True,
    torch_dtype=torch.float16
)
print("Loading better VAE")
better_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)
print("Loading sdxl")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=better_vae,
    controlnet=controlnet,
    use_safetensors=True,
    variant="fp16",
    torch_dtype=torch.float16,
)
#pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()


t2 = time.time()
print("Setup took: ", t2 - t1)


def get_depth_map(image):
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

prompt = "stormtrooper lecture, photorealistic"
image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
controlnet_conditioning_scale = 0.5  # recommended for good generalization

depth_image = get_depth_map(image)

#depth_image.save(f"depthstormtrooper.png")
#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
images = pipe(prompt, image=depth_image, num_inference_steps=30, controlnet_conditioning_scale=controlnet_conditioning_scale,).image[0]
#images[0].save(f"stormtrooper.png")