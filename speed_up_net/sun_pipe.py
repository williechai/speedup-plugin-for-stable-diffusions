from typing import Union, List, Optional
import torch
import PIL
import numpy as np
import os
import logging
import json
#from .attention_processor import is_torch2_available
from peft import LoraModel, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict 
from safetensors.torch import load_file as safe_load

import torch.nn.functional as F
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

if is_torch2_available():
    from .attention_processor import AttnProcessor2_0 as AttnProcessor
else:
    from .attention_processor import AttnProcessor

from .attention_processor import SUNAttnProcessor
from IPython import embed

from diffusers import (
    StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
)

def _build_adapter(unet, ip_param_scale=False, add_pos=False):
    attn_procs = {}

    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = SUNAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=77, param_scale=ip_param_scale, add_pos=add_pos)        

    unet.set_attn_processor(attn_procs)
    return unet

def _load_adapter(adapt_unet, adapter_path):
    if adapter_path.endswith("safetensors"):
        state_dict = safe_load(adapter_path, "cpu")
    else:
        state_dict = torch.load(adapter_path, map_location='cpu')
    adapter_modules = torch.nn.ModuleList(adapt_unet.attn_processors.values())
    adapter_modules.load_state_dict(state_dict)
    return adapt_unet


def _load_lora_dir_format(pipe, lora_dir):    
    torch_device = pipe.unet.device
    torch_dtype = pipe.unet.dtype  

    loraModelPath = os.path.join(lora_dir, "lora.pt")
    loraJsonPath = os.path.join(lora_dir, "lora_config.json")

    if not os.path.exists(loraJsonPath) or not os.path.exists(loraModelPath):
        logging.error(f"{lora_dir}: lora file not exists")
        exit(27)

    with open(loraJsonPath, "r") as f:
        lora_config = json.load(f)

    lora_checkpoint = torch.load(loraModelPath, map_location="cpu")

    unet_lora = {k: v for k, v in lora_checkpoint.items() if "text_encoder_" not in k}
    text_encoder_lora = {k.replace("text_encoder_", ""): v for k, v in lora_checkpoint.items() if "text_encoder_" in k}
    
    unet_config = LoraConfig(**lora_config["peft_config"])
    pipe.unet = LoraModel(unet_config, pipe.unet).to(torch_device).to(torch_dtype)
    set_peft_model_state_dict(pipe.unet, unet_lora)

    if "text_encoder_peft_config" in lora_config:
        text_encoder_config = LoraConfig(**lora_config["text_encoder_peft_config"])
        pipe.text_encoder = LoraModel(text_encoder_config, pipe.text_encoder).to(torch_device).to(torch_dtype)
        set_peft_model_state_dict(pipe.text_encoder, text_encoder_lora)
    

class SUNPipe:
    def __init__(self, sd_pipe, sun_adapter_path, add_pos=False):
        self.pipe = sd_pipe
        unet = self.pipe.unet
        torch_device = unet.device
        torch_dtype = unet.dtype  
        unet = _build_adapter(unet, ip_param_scale=False, add_pos=add_pos).to(torch_device, torch_dtype)
        unet = _load_adapter(unet, sun_adapter_path).to(torch_device, torch_dtype)
        self.pipe.unet = unet
        self.torch_device = torch_device
        self.pipe.scheduler.register_to_config(timestep_spacing="trailing")

    def load_lora_weights(self, lora_path):
        if lora_path.endswith(".safetensors"):
            self.pipe.load_lora_weights(lora_path)
        elif os.path.isdir(lora_path):
            pt_path = os.path.join(lora_path, "lora.pt")
            json_path = os.path.join(lora_path, "lora_config.json")
            assert os.path.exists(pt_path)
            assert os.path.exists(json_path)
            #assert False, "not supported yet"
            _load_lora_dir_format(self.pipe, lora_path)
        else:
            raise ValueError("lora path unrecognized")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        mask_image: Union[torch.Tensor, PIL.Image.Image] = None,
        control_image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.5,
        eta: float = 0.0,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            bsz = 1
        elif prompt is not None and isinstance(prompt, list):
            bsz = len(prompt)
        else:
            bsz = prompt_embeds.shape[0]

        with torch.inference_mode():
            pos_neg_embed_cat_in_dim0 = self.pipe._encode_prompt(
                prompt,
                device=self.torch_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds
            )
            assert pos_neg_embed_cat_in_dim0.size(0) == bsz * 2
            neg_embed, pos_embed = pos_neg_embed_cat_in_dim0.chunk(2, dim=0)
            pos_neg_embed_cat_in_dim1 = torch.cat([pos_embed, neg_embed], dim=1)
            assert pos_neg_embed_cat_in_dim1.size(0) == bsz 
            assert pos_neg_embed_cat_in_dim1.size(1) == 77 * 2
            
            if isinstance(self.pipe, StableDiffusionControlNetInpaintPipeline):             
                #"""
                images = self.pipe(
                    image=image,
                    mask_image=mask_image,
                    control_image=control_image,
                    height=height,
                    width=width,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    prompt_embeds=pos_neg_embed_cat_in_dim1,
                    guidance_scale=0.0,
                    generator=generator,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    eta=eta
                )
        
            elif isinstance(self.pipe, StableDiffusionInpaintPipeline):
                assert control_image is None
                images = self.pipe(
                    image=image,
                    mask_image=mask_image,
                    height=height,
                    width=width,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    prompt_embeds=pos_neg_embed_cat_in_dim1,
                    guidance_scale=0.0,
                    generator=generator,
                    eta=eta
                )
            elif isinstance(self.pipe, StableDiffusionControlNetPipeline):
                assert control_image is None
                assert mask_image is None
                images = self.pipe(
                    image=image,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    prompt_embeds=pos_neg_embed_cat_in_dim1,
                    guidance_scale=0.0,
                    generator=generator,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    eta=eta
                )
            elif isinstance(self.pipe, StableDiffusionPipeline):
                assert control_image is None
                assert mask_image is None
                assert image is None
                images = self.pipe(
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    prompt_embeds=pos_neg_embed_cat_in_dim1,
                    guidance_scale=0.0,
                    generator=generator,
                    eta=eta
                )
            elif isinstance(self.pipe, StableDiffusionImg2ImgPipeline):
                assert control_image is None
                assert mask_image is None
                assert image is not None
                images = self.pipe(
                    image=image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    prompt_embeds=pos_neg_embed_cat_in_dim1,
                    guidance_scale=0.0,
                    generator=generator,
                    eta=eta
                )
                
            else:
                raise ValueError("not supported {}".format(type(self.pipe)))

        return images
    


    