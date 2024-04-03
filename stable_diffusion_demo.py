import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A vibrant watercolor painting of a majestic lion in the savanna"
image = pipe(prompt).images[0]  
image.save("lion_watercolor.png") 