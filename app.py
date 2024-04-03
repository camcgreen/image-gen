# from flask import Flask
# app = Flask(__name__)

# @app.route("/")
# def home():
#     return "Hello, World!"

import torch
from diffusers import StableDiffusionPipeline
from flask import Flask

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

app = Flask(__name__)

@app.route('/')
def home():
    return "Stable Diffusion Image Generator"  

@app.route('/generate')
def generate_image():
    prompt = "A vibrant watercolor painting of a majestic lion in the savanna" # Hardcoded prompt 
    image = pipe(prompt).images[0]
    image_filename = "generated_image.png"
    image.save(image_filename)
    return "Image saved as " + image_filename 

if __name__ == '__main__':
    app.run(debug=True) 
