import torch
from diffusers import StableDiffusionPipeline
from flask import Flask, request

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

app = Flask(__name__)

@app.route('/')
def home():
    return "Stable Diffusion Image Generator"  

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.form.get('prompt')  # Get the prompt from the request
    if prompt:
        image = pipe(prompt).images[0]
        image_filename = "generated_image.png"
        image.save(image_filename)
        return "Image saved as " + image_filename 
    else:
        return "Please provide a text prompt."

if __name__ == '__main__':
    app.run(debug=True) 
