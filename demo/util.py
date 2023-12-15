import os
from speed_up_net.sun_pipe import SUNPipe

from diffusers import DDIMScheduler, UNet2DConditionModel, ControlNetModel
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionControlNetPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import cv2
import numpy as np 

import torch


def get_stable_diffusion_pipe(
        base_model_path, 
        unet_model_path=None,
        scheluler_cls=DDIMScheduler
    ):

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path, safety_checker=None)
    pipe.scheduler = scheluler_cls.from_config(pipe.scheduler.config)
    pipe.scheduler.register_to_config(timestep_spacing="trailing")

    torch_dtype = pipe.unet.dtype
    torch_device = pipe.unet.device
    if unet_model_path:
        unet = UNet2DConditionModel.from_pretrained(
            unet_model_path, subfolder="unet", 
            use_safetensors=(False if os.path.exists(os.path.join(unet_model_path, "unet", "diffusion_pytorch_model.bin")) else True)
        ).to(torch_dtype).to(torch_device)
        pipe.unet = unet

    return pipe

def get_stable_diffusion_img2img(
        base_model_path, 
        unet_model_path=None,
        scheluler_cls=DDIMScheduler
    ):

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_path, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = scheluler_cls.from_config(pipe.scheduler.config)
    pipe.scheduler.register_to_config(timestep_spacing="trailing")
    pipe.vae.to(torch.float16)

    torch_dtype = pipe.unet.dtype
    torch_device = pipe.unet.device
    if unet_model_path:
        unet = UNet2DConditionModel.from_pretrained(
            unet_model_path, subfolder="unet", 
            use_safetensors=(False if os.path.exists(os.path.join(unet_model_path, "unet", "diffusion_pytorch_model.bin")) else True)
        ).to(torch_dtype).to(torch_device)
        pipe.unet = unet

    return pipe

def get_stable_diffusion_controlnet_pipe(
        base_model_path, 
        controlnet_model_path,
        unet_model_path=None,
        scheluler_cls=DDIMScheduler
    ):
    #controlnet_model_path = os.path.join(model_root, "models--lllyasviel--control_v11p_sd15_openpose/")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_path
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, safety_checker=None)
    
    pipe.to(torch.float16)

    pipe.scheduler = scheluler_cls.from_config(pipe.scheduler.config)
    pipe.scheduler.register_to_config(timestep_spacing="trailing")

    torch_dtype = pipe.unet.dtype
    torch_device = pipe.unet.device
    if unet_model_path:
        unet = UNet2DConditionModel.from_pretrained(
            unet_model_path, subfolder="unet", 
            use_safetensors=(False if os.path.exists(os.path.join(unet_model_path, "unet", "diffusion_pytorch_model.bin")) else True)
        ).to(torch_dtype).to(torch_device)
        pipe.unet = unet

    return pipe

def get_stable_diffusion_inpainting_pipe(
        base_model_path, 
        scheluler_cls=DDIMScheduler
    ):

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model_path, safety_checker=None)
    pipe.scheduler = scheluler_cls.from_config(pipe.scheduler.config)
    pipe.scheduler.register_to_config(timestep_spacing="trailing")

    #torch_dtype = pipe.unet.dtype
    #torch_device = pipe.unet.device   
    return pipe

def pil_up_crop_square(img):
    size = img.size
    short_size = size[1] if size[0] > size[1] else size[0]
    crop_left = (size[0] - short_size) // 2
    crop_right = crop_left + short_size
    crop_up = 0
    crop_down = crop_up + short_size
    return img.crop((crop_left, crop_up, crop_right, crop_down))

def pil_center_crop_square(img, ratio=1.0):
    size = img.size
    short_size = size[1] if size[0] > size[1] else size[0]
    center_size = int(short_size * ratio)
    crop_left = (size[0] - center_size) // 2
    crop_right = crop_left + center_size
    crop_up = (size[1] - center_size) // 2
    crop_down = crop_up + center_size
    return img.crop((crop_left, crop_up, crop_right, crop_down))

def cv2_center_crop_square(img, ratio=1.0):
    size = (img.shape[1], img.shape[0])
    short_size = size[1] if size[0] > size[1] else size[0]
    center_size = int(short_size * ratio)
    crop_left = (size[0] - center_size) // 2
    crop_right = crop_left + center_size
    crop_up = (size[1] - center_size) // 2
    crop_down = crop_up + center_size
    #return img.crop((crop_left, crop_up, crop_right, crop_down))
    return img[crop_up:crop_down, crop_left:crop_right]

def tensor_to_pil(tensor):
    image = (tensor / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    return image

def tensor_to_cv2(tensor):
    image = ((tensor / 2 + 0.5).clamp(0, 1) * 255).squeeze().to(torch.uint8)
    image = image.round().cpu().permute(1, 2, 0).numpy()
    image = image[:,:,::-1]
    return image

def cv2pil(mat):
    return Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))

def pil2cv2(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)

def pil_to_tensor(img):
    return torch.from_numpy((np.asarray(img).astype(np.float32) / 255)).permute(2, 0, 1)
    


