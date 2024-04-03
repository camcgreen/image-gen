import os
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from flask import Flask, request

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_2step_unet.safetensors"

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

app = Flask(__name__)

@app.route('/')
def home():
    return "Stable Diffusion Image Generator"  

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.form.get('prompt')

    if prompt:
        documents_path = os.path.join(os.path.expanduser('~'), 'Documents')
        sd_folder = os.path.join(documents_path, 'sd')
        os.makedirs(sd_folder, exist_ok=True)  
        image = pipe(prompt).images[0]
        # Ensure using the same inference steps as the loaded model and CFG set to 0.
        image = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
        image_filename = "generated_image.png"
        image_path = os.path.join(sd_folder, image_filename)  
        image.save(image_path)
        return "Image saved in your Documents folder (inside the 'sd' folder)" 
    else:
        return "Please provide a text prompt."

if __name__ == '__main__':
    app.run(debug=True) 
