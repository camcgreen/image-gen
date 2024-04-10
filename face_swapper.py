import base64
import io
import os

import cv2
import insightface
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from matplotlib.figure import Figure


class face_swapper:
    def __init__(this):
        assert float(".".join(insightface.__version__.split(".")[:2])) >= float("0.7")
        this.app = FaceAnalysis(name="buffalo_l")
        this.app.prepare(ctx_id=0, det_size=(640, 640))
        this.swapper = insightface.model_zoo.get_model(
            "./inswapper_128.onnx", download=False, download_zip=False
        )

    def swap_face_from_file(this, input_face_path, input_image_path):
        print("Received images")

        input_face_base64 = input_face_path
        print("Input face converted to base64")

        input_image_base64 = input_image_path
        print("Input image converted to base64")
        print("Received images")

        # Add padding if needed
        while len(input_face_base64) % 4 != 0:
            input_face_base64 += "="

        # Convert input face from base64 to image
        try:
            input_face = np.frombuffer(
                base64.b64decode(input_face_base64), dtype=np.uint8
            )
            input_face = cv2.imdecode(input_face, cv2.IMREAD_COLOR)
            print("Input face decoded")
        except Exception as e:
            print("Error decoding input face:", str(e))
            return None

        # Convert input image from base64 to image
        try:
            input_image = np.frombuffer(
                base64.b64decode(input_image_base64), dtype=np.uint8
            )
            input_image = cv2.imdecode(input_image, cv2.IMREAD_COLOR)
            print("Input image decoded")
        except Exception as e:
            print("Error decoding input image:", str(e))
            return None

        # Set input face
        img = input_face
        print("Face detected")

        # Detect faces on the input face image
        faces_on_input_face_img = this.app.get(img)
        frame_width = 768
        frame_height = 1080
        print("Faces detected on input face image")
        distances = []
        frame_center = np.array([frame_width / 2, frame_height / 2])
        for face in faces_on_input_face_img:
            face_center = np.array(
                [(face.bbox[2] + face.bbox[0]) / 2, (face.bbox[3] + face.bbox[1]) / 2]
            )
            distance = np.linalg.norm(face_center - frame_center)
            print("distance", distance)
            distances.append(distance)

        target_index = np.argmin(distances)
        # Sets which face we want to use from the faces that have been detected on the input face image
        # source_face = faces_on_input_face_img[0]
        print("number of faces detected = ", len(faces_on_input_face_img))
        print("target_index = ", target_index)
        source_face = faces_on_input_face_img[target_index]

        # Sets image to be mutated
        new_img = input_image

        # Detect faces on the image
        faces_input_img = this.app.get(new_img)
        print("Face detected on input image")

        res = new_img.copy()
        res = this.swapper.get(res, faces_input_img[0], source_face, paste_back=True)
        print("Face mutated")

        res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        print("Image converted to RGB")

        fig = Figure()
        ax = fig.subplots()

        ax.axis("off")
        ax.imshow(res_rgb)

        # plt.margins(x=0)
        # plt.margins(y=0)

        # fig.savefig('output.png', dpi=300, bbox_inches='tight')
        fig.savefig("output.png", dpi=300, bbox_inches="tight", pad_inches=0)

        # Convert output image to base64
        output_img_base64 = ""
        output_path = "./output.png"
        # Read input face image from file
        with open(output_path, "rb") as f:
            output_img_base64 = f.read()
        input_face_base64 = base64.b64encode(output_img_base64).decode("utf-8")
        print("Input face converted to base64")

        return input_face_base64
