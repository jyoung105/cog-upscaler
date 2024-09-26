# Flux

SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis

## Reference

- [project](https://stability.ai/news/stable-diffusion-sdxl-1-announcement)
- [arxiv](https://arxiv.org/abs/2307.01952)
- [github](https://github.com/Stability-AI/generative-models)
- [hugging face-base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [hugging face-refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Normal/Stable/SDXL
```

3. download weights before deployment
```
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```