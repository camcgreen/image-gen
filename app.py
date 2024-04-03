import os
import torch
from diffusers import StableDiffusionXLPipeline
from flask import Flask, request

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

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
        image_filename = "generated_image.png"
        image_path = os.path.join(sd_folder, image_filename)  
        image.save(image_path)
        return "Image saved in your Documents folder (inside the 'sd' folder)" 
    else:
        return "Please provide a text prompt."

if __name__ == '__main__':
    app.run(debug=True) 
