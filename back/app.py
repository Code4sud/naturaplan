from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html") 

@app.route("/GetImage", methods=["GET"])
def GetImage():
    return render_template("Generate.html")

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.form.get("prompt")
    file = request.files.get("image")

    if not prompt or not file:
        return jsonify({"error": "Un prompt et une image doivent être fournis"}), 400

    image = Image.open(file)

    generated_image = pipe(prompt, init_image=image).images[0]

    image_path = "static/images/generated_image.png"
    generated_image.save(image_path)

    return jsonify({"message": "Image générée avec succès", "image_path": "/" + image_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
