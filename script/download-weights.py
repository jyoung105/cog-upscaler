from diffusers import ControlNetModel, DiffusionPipeline, AutoencoderKL
import torch
from RealESRGAN import RealESRGAN


for scale in [2, 4]:
    model = RealESRGAN("cuda", scale=scale)
    model.load_weights(f"weights/RealESRGAN_x{scale}.pth", download=True)

SD15_WEIGHTS = "weights"
CONTROLNET_CACHE = "controlnet-cache"

controlnet = ControlNetModel.from_single_file(
    "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/blob/main/bdsqlsz_controlllite_xl_tile_realistic.safetensors", torch_dtype=torch.float16, cache_dir=CONTROLNET_CACHE
)
controlnet.save_pretrained(CONTROLNET_CACHE)

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")

pipe = DiffusionPipeline.from_single_file(
    "https://huggingface.co/dataautogpt3/TempestV0.1/blob/main/TempestV0.1-Artistic.safetensors", torch_dtype=torch.float16, cache_dir=SD15_WEIGHTS, vae=vae
)
pipe.save_pretrained(SD15_WEIGHTS)
