# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import List

import os

import gc
import time

import torch
import torch.nn.functional as F
import numpy as np

import cv2
import PIL
from PIL import Image

from diffusers.models import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from diffusers.utils import load_image, logging
from diffusers.utils.logging import set_verbosity

from safetensors.torch import load_file


# Remove warning messages
set_verbosity(logging.ERROR)


# GPU global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if str(DEVICE).__contains__("cuda") else torch.float32


# AI global variables
TOTAL_CACHE = "./cache"
MODEL_CACHE = "./flux-cache"

MODEL_ID = "black-forest-labs/FLUX.1-dev"
UPSCALER_ID = "jasperai/Flux.1-dev-Controlnet-Upscaler"
HYPER_ID = "ByteDance/Hyper-SD"
HYPER_FILE = "Hyper-FLUX.1-dev-8steps-lora.safetensors"


# Set safety checker
# SAFETY_CACHE = "./safetys"
# FEATURE_EXTRACTOR = "./feature-extractors"
# SAFETY_URL = "https://weights.replicate.delivery/default/playgroundai/safety-cache.tar"


def flush():
    gc.collect()
    torch.cuda.empty_cache()


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.load_flux()


    def load_sdxl(self):
        print("[~] Setup pipeline")
        # 1. Setup pipeline
        self.controlnet = FluxControlNetModel.from_pretrained(
            TOTAL_CACHE,
            torch_dtype=torch.bfloat16
        )
        self.pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=self.controlnet,
            torch_dtype=torch.bfloat16,
        )
        
        
        # 2. Setup IP-Adapter 
        # self.pipe.load_ip_adapter(
        #     ["ostris/ip-composition-adapter", "h94/IP-Adapter"],
        #     subfolder=["", "sdxl_models"],
        #     weight_name=[
        #         "ip_plus_composition_sdxl.safetensors",
        #         "ip-adapter_sdxl_vit-h.safetensors",
        #     ],
        #     image_encoder_folder=None,
        # ) 
        # self.pipe.load_ip_adapter(
        #     ["h94/IP-Adapter"],
        #     subfolder=["sdxl_models"],
        #     weight_name=[
        #         "ip-adapter-plus_sdxl_vit-h.safetensors",
        #     ],
        #     image_encoder_folder=None,
        # )
        
        # self.pipe = self.pipe.to(DEVICE)
        
        
        # 3. Enable to use FreeU
        # self.pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
        
        
        # 4. Add textual inversion
        # embedding_1 = load_file(f"{TOTAL_CACHE}/ac_neg1.safetensors")
        # embedding_2 = load_file(f"{TOTAL_CACHE}/ac_neg2.safetensors")
        
        # self.pipe_prev.load_textual_inversion(embedding_1["clip_l"], token="<ac_neg1>", text_encoder=self.pipe_prev.text_encoder, tokenizer=self.pipe_prev.tokenizer)
        # self.pipe_prev.load_textual_inversion(embedding_1["clip_g"], token="<ac_neg1>", text_encoder=self.pipe_prev.text_encoder_2, tokenizer=self.pipe_prev.tokenizer_2)
        # self.pipe_prev.load_textual_inversion(embedding_2["clip_l"], token="<ac_neg2>", text_encoder=self.pipe_prev.text_encoder, tokenizer=self.pipe_prev.tokenizer)
        # self.pipe_prev.load_textual_inversion(embedding_2["clip_g"], token="<ac_neg2>", text_encoder=self.pipe_prev.text_encoder_2, tokenizer=self.pipe_prev.tokenizer_2)
        
        
        # 5. Add LoRA
        # self.pipe.load_lora_weights(hf_hub_download("jyoung105/general-lora", "add-detail-xl.safetensors"), adapter_name="<add_detail>")
        # self.pipe.load_lora_weights(hf_hub_download("jyoung105/general-lora", "sd_xl_offset_example-lora_1.0.safetensors"), adapter_name="<noise_offset>")
        # self.pipe.load_lora_weights(hf_hub_download("jyoung105/general-lora", "xl_more_art-full_v1.safetensors"), adapter_name="<art_full>")

        # self.pipe.set_adapters(["<add_detail>", "<noise_offset>", "<art_full>"], adapter_weights=[0.5, 0.5, 0.5])
        
        self.pipe.load_lora_weights(TOTAL_CACHE, weight_name=HYPER_FILE, local_files_only=True)
        self.pipe.fuse_lora(lora_scale=0.125)
        
        self.pipe = self.pipe.to(DEVICE)
        
        # 6. Optimization
        # Inference speed
        # self.pipe.enable_vae_slicing()
        # self.pipe.enable_vae_tiling()
        # self.pipe.enable_attention_slicing()

        # Memory
        # self.pipe.enable_model_cpu_offload() # This optimization slightly reduces memory consumption, but is optimized for speed.
        # self.pipe.enable_sequential_cpu_offload() # This optimization reduces memory consumption, but also reduces speed.
        # self.pipe.enable_xformers_memory_efficient_attention() # useless for torch > 2.0, but if using torch < 2.0, this is an essential optimization.
        
        # PyTorch
        # self.pipe.unet.set_attn_processor(AttnProcessor2_0())
        # self.pipe.fuse_qkv_projections()
        # self.pipe.unet.to(memory_format=torch.channels_last)
        # self.pipe.vae.to(memory_format=torch.channels_last)
        
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True) # max-autotune or reduce-overhead
        # self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode="reduce-overhead", fullgraph=True)
        # self.pipe.upcast_vae()


    @torch.inference_mode()
    def generate_image(
        self,
        control_image,
        upscale_factor,
        prompt,
        negative_prompt,
        num_outputs,
        num_steps,
        guidance_scale,
        controlnet_scale,
        seed,
    ):
        flush()
        print(f"[Debug] Prompt: {prompt}")
        
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        width, height = control_image.size
        control_image = control_image.resize((width * upscale_factor, height * upscale_factor))
        
        image_list = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_outputs,
            num_inference_steps=num_steps,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_scale,
            width=control_image.size[0],
            height=control_image.size[1],
            generator=generator,
        ).images
        
        return image_list


    @torch.inference_mode()
    def predict(
        self,
        control_image: Path = Input(
            description="Input image for upscale.",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt, text what you want to put on.",
            default=None,
        ),
        negative_prompt: str = Input(
            description="Input negative prompt, text what you don't want to put on.",
            default=None,
        ),
        upscale_factor: int = Input(
            description="Scale how much you want to upscale an image.",
            default=2,
            choices=[2, 4],
        ),
        num_images: int = Input(
            description="Number of outputs.",
            default=1,
            ge=1,
            le=4,
        ),
        steps: int = Input(
            description="Number of denoising steps.",
            default=8,
            ge=1,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance.",
            default=3.5,
            ge=0,
            le=20,
        ),
        controlnet_scale: float = Input(
            description="Scale for controlnet conditioning.",
            default=0.6,
            ge=0,
            le=1,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.",
            default=None,
        ),
        # output_type: str = Input(
        #     description="Format of the output",
        #     default="webp",
        #     choices=["png", "jpg", "webp"]
        # ),
        # output_quality: int = Input(
        #     description="Quality of the output",
        #     default=80,
        #     ge=0,
        #     le=100,
        # ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        start1 = time.time() # stamp time
        
        if control_image is None:
            msg = "No image, Save money"
            return msg

        else:
            print(f"DEVICE: {DEVICE}")
            print(f"DTYPE: {DTYPE}")
            
            # If no seed is provided, generate a random seed
            if seed is None:
                seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")

            # Set prompt and negative_prompt
            if prompt is None:
                prompt = ""
            
            if negative_prompt is None:
                negative_prompt = ""

            new_prompt = prompt + "best quality, high detail, sharp focus"
            new_negative_prompt = negative_prompt
            
            control_image = load_image(control_image)
            
            print("Finish setup in " + str(time.time()-start1) + " secs.")

            start2 = time.time() # stamp time
            
            base_image = self.generate_image(
                control_image=control_image,
                upscale_factor=upscale_factor,
                prompt=new_prompt,
                negative_prompt=new_negative_prompt,
                num_outputs=num_images,
                num_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_scale=controlnet_scale,
                seed=seed,
            )
            print("Finish generation in " + str(time.time()-start2) + " secs.")

            # Save the generated images
            # TODO : check for NSFW content
            output_paths = []
            for i, image in enumerate(base_image):
                output_path = f"/tmp/out_{i}.png"
                image.save(output_path)
                output_paths.append(Path(output_path))
            
            return output_paths