import base64
import io
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
from PIL import Image
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


@app.route("/generate", methods=["POST"])
def generate_image():
    prompt = request.form.get("prompt")
    face_image_data = request.form.get("face_image")

    if prompt:
        print("starting image gen")
        image = pipe(prompt, num_inference_steps=2, guidance_scale=0).images[0]
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            back_image_data = base64.b64encode(output.getvalue()).decode("ascii")

        try:
            print("starting face swap")
            print("req detected")
            output_image_base64 = swapper_obj.swap_face_from_file(
                face_image_data, back_image_data
            )
            return output_image_base64

        except KeyError:
            return jsonify({"error": "Invalid request. Missing image field."}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return "Please provide a text prompt."


if __name__ == "__main__":
    app.run(port=6969)
