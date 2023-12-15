import torch
from diffusers import DPMSolverMultistepScheduler
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
from IPython import embed
from diffusers.image_processor import VaeImageProcessor
from diffusers import ControlNetModel
import time
from IPython import embed
from controlnet_aux import CannyDetector, OpenposeDetector
import cv2

import sys
sys.path.insert(0, '..')
from speed_up_net.sun_pipe import SUNPipe

from util import get_stable_diffusion_controlnet_pipe, pil_up_crop_square


weight_dtype = torch.float16
torch_device = "mps"

# model for base model
model_path = "../models/sd_v15"
# model for style unet
# https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE
style_unet_model_path = "../models/Realistic_Vision_V5.1_noVAE"
# model for controlnet
controlnet_model_path = "../models/sd-controlnet-canny"

pipe = get_stable_diffusion_controlnet_pipe(
    base_model_path=model_path, 
    controlnet_model_path=controlnet_model_path,
    unet_model_path=None, #style_unet_model_path
    scheluler_cls=DPMSolverMultistepScheduler,
).to(torch_device)

# Refer to README.md to Download 
adapter_path = "../test_cache/sun_adapter_sdv15_4step_addpos.safetensors" 
pipe = SUNPipe(pipe, adapter_path, add_pos=True)

# https://civitai.com/models/24833/minimalist-anime-style
lora_path = "../models/anime_minimalist_v1-000020.safetensors"

if lora_path:
    pipe.load_lora_weights(lora_path)

canny_detecter = CannyDetector()

img_pil = pil_up_crop_square(Image.open("../pics/boy.png")).resize((512, 512))

default_negative_prompt = "worst quality, low quality, blurry, bad hand, watermark, multiple limbs, deformed fingers, bad fingers, ugly, monochrome, horror, geometry, bad anatomy, bad limbs, Blurry pupil, bad shading, error, bad composition, Extra fingers, strange fingers, Extra ears, extra leg, bad leg, disability, Blurry eyes, bad eyes, Twisted body, confusion, bad legs"
prompt = "anime minimalist, 1 20 y.o man, solo, closeup face photo in sweater, cleavage, pale skin"
negative_prompt = default_negative_prompt

#anime minimalist, anime minimalist, 


with torch.no_grad(): 
    ctn_img = canny_detecter(img_pil)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=4,
        generator=torch.manual_seed(0),
        eta=0.0,
        image=ctn_img,  
        controlnet_conditioning_scale=0.75,
    ).images[0]


    image.save("anime_lora_sun.jpg")
