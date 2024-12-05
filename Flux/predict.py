# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

# Standard library imports
import os
import random
import gc
import time
from typing import List

# Third-party library imports
import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image

from diffusers.utils import logging
from diffusers.utils.logging import set_verbosity

from safetensors.torch import load_file

import GPUtil

os.chdir("/content/ComfyUI")
import nodes
from nodes import NODE_CLASS_MAPPINGS

from comfy_extras import nodes_custom_sampler, nodes_flux, nodes_model_advanced, nodes_upscale_model
from comfy import model_management

# Cog imports
from cog import BasePredictor, Input, Path

# Set logging level
set_verbosity(logging.ERROR)

# GPU global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32  # bf16 or fp32


def get_gpu_details() -> List:
    """Retrieve GPU details using GPUtil."""
    gpus = GPUtil.getGPUs()
    gpu_details = []
    for gpu in gpus:
        gpu_info = {
            "id": gpu.id,
            "name": gpu.name,
            "load": f"{gpu.load * 100:.2f}",  # GPU load in percentage
            "memory_free": gpu.memoryFree,
            "memory_used": gpu.memoryUsed,
            "memory_total": gpu.memoryTotal,
            "temperature": gpu.temperature,
            "uuid": gpu.uuid
        }
        gpu_details.append(gpu_info)
    return gpu_details


def add_memory_unit(mem: float) -> str:
    """Convert memory size to a human-readable format with appropriate units."""
    if mem > 1024:
        mem = round(mem / 1024, 2)
        mem = f"{mem} GiB"
    else:
        mem = round(mem, 2)
        mem = f"{mem} MiB"
    return mem


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)                          # Set the Python built-in random seed
    np.random.seed(seed)                       # Set the NumPy random seed
    torch.manual_seed(seed)                    # Set the PyTorch random seed for CPU
    torch.cuda.manual_seed_all(seed)           # Set the PyTorch random seed for all GPUs
    torch.backends.cudnn.benchmark = False     # Disable CUDNN benchmark for deterministic behavior
    torch.backends.cudnn.deterministic = True  # Ensure deterministic CUDNN operations


def flush() -> None:
    """Clear GPU cache."""
    torch.cuda.synchronize() # Synchronize CUDA operations
    gc.collect()             # Collect garbage
    torch.cuda.empty_cache() # Empty CUDA cache


def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the Flux pipeline and related components."""
        self.load_flux()


    def load_flux(self):
        print("[~] Setup pipeline")
        self.DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        self.RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
        self.BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
        self.KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        self.BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
        self.SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        self.FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()
        self.ModelSamplingFlux = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
        self.VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
        self.VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        self.EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        self.LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.UpscaleModelLoader = nodes_upscale_model.NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        
        from quantized.nodes import NODE_CLASS_MAPPINGS
        self.UNETLoader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        
        from upscale.nodes import NODE_CLASS_MAPPINGS
        self.UltimateSDUpscale = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]()
        
        with torch.inference_mode():
            self.clip = self.DualCLIPLoader.load_clip("t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux")[0]
            self.unet = self.UNETLoader.load_unet("hyper-flux-8step-Q4_0.gguf")[0]
            self.vae = self.VAELoader.load_vae("ae.sft")[0]
            self.upscaler = self.UpscaleModelLoader.load_model("4x_NMKD-Superscale-SP_178000_G.pth")[0]


    @torch.inference_mode()
    def generate_image(
        self,
        image,
        upscale_factor,
        prompt,
        num_steps,
        guidance_scale,
        denoise_scale,
        seed,
    ):
        flush()
        print(f"[Debug] Prompt: {prompt}")
        print(f"[Debug] Seed: {str(seed)}")
        
        # Prompt
        cond, pooled = self.clip.encode_from_tokens(self.clip.tokenize(prompt), return_pooled=True)
        cond = [[cond, {"pooled_output": pooled}]]
        sampler = self.ModelSamplingFlux.patch(self.unet, 1.15, 0.5, width, height)[0] # model
        
        # Image
        init_image = self.LoadImage.load_image(image)[0] # image
        width, height = image.size
        tile_width = width * upscale_factor / 2 + 32
        tile_height = height * upscale_factor / 2 + 32
        
        # Guidance
        guider = self.FluxGuidance.append(cond, guidance_scale)[0] # positive
        neg_guider = self.ConditioningZeroOut.zero_out(cond)[0] # negative
        
        model_management.soft_empty_cache()
        
        upscale_image = self.UltimateSDUpscale.upscale(
            image=init_image,
            model=sampler,
            positive=guider,
            negative=neg_guider,
            vae=self.vae,
            upscale_by=upscale_factor,
            seed=seed,
            steps=num_steps,
            cfg=1.0,
            sampler_name="dpmpp_2m",
            scheduler="beta",
            denoise=denoise_scale,
            upscale_model=self.upscaler,
            mode_type="Linear", 
            tile_width=tile_width,
            tile_height=tile_height,
            mask_blur=16, 
            tile_padding=32,
            seam_fix_mode=None, 
            seam_fix_denoise=0.25, 
            seam_fix_mask_blur=16,
            seam_fix_width=64, 
            seam_fix_padding=16, 
            force_uniform_tiles=True, 
            tiled_decode=True
        )
        image_list = [to_pil_image(upscale_image[0])]
        
        # decoded = self.VAEDecode.decode(self.vae, sample)[0].detach()
        # image_list = [Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0])]
        
        return image_list


    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image for upscale.",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt, text what you want to put on.",
            default=None,
        ),
        upscale_factor: float = Input(
            description="Scale how much you want to upscale an image.",
            default=2.0,
            choices=[2.0, 4.0],
        ),
        steps: int = Input(
            description="Number of denoising steps.",
            default=8,
            ge=1,
            le=30,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance.",
            default=3.5,
            ge=0,
            le=20,
        ),
        denoise_scale: float = Input(
            description="Scale for denoising.",
            default=0.2,
            ge=0,
            le=1,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.",
            default=None,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        start1 = time.time() # stamp time
        
        if image is None:
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

            new_prompt = prompt + "best quality, high detail, sharp focus"
            
            print("Finish setup in " + str(time.time()-start1) + " secs.")

            start2 = time.time() # stamp time
            
            base_image = self.generate_image(
                image=image,
                upscale_factor=upscale_factor,
                prompt=new_prompt,
                num_steps=steps,
                guidance_scale=guidance_scale,
                denoise_scale=denoise_scale,
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