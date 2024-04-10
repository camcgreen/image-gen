import os

import torch
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from flask import Flask, jsonify, request
from flask_cors import CORS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from face_swapper import face_swapper

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_2step_unet.safetensors"

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
    "cuda", torch.float16
)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(
    base, unet=unet, torch_dtype=torch.float16, variant="fp16"
).to("cuda")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)

app = Flask(__name__)
CORS(app)
swapper_obj = face_swapper()

print("starting server...")


@app.route("/")
def home():
    return "Stable Diffusion Image Generator"


@app.route("/process_image", methods=["POST"])
def process_image():
    try:
        print("req detected")
        face_image_data = request.json["face_image"]
        back_image_data = request.json["back_image"]
        print("images on base64 received")
        output_image_base64 = swapper_obj.swap_face_from_file(
            face_image_data, back_image_data
        )
        return output_image_base64

    except KeyError:
        return jsonify({"error": "Invalid request. Missing image field."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate_image():
    prompt = request.form.get("prompt")

    if prompt:
        image = pipe(prompt, num_inference_steps=2, guidance_scale=0).images[0]
        return "image successfully generated with prompt " + prompt
        # documents_path = os.path.join(os.path.expanduser('~'), 'Documents')
        # sd_folder = os.path.join(documents_path, 'sd')
        # os.makedirs(sd_folder, exist_ok=True)
        # # Ensure using the same inference steps as the loaded model and CFG set to 0.
        # image_filename = "generated_image.png"
        # image_path = os.path.join(sd_folder, image_filename)
        # image.save(image_path)
        # return "Image saved in your Documents folder (inside the 'sd' folder)"
    else:
        return "Please provide a text prompt."


if __name__ == "__main__":
    app.run(port=6969)
