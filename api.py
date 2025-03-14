from flask import Flask, request, jsonify
import torch
import os
import yaml
from PIL import Image
import io
import base64
import torchvision.transforms as transforms
from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor

app = Flask(__name__)

# Load config and models
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
device = config.get("device", "cpu")

cnn_model = CNNDragPredictor(latent_dim=config["cnn"].get("latent_dim", 64)).to(device)
cnn_model.load_state_dict(torch.load(os.path.join("models","cnn_drag_predictor.pth"), map_location=device))
cnn_model.eval()

@app.route("/predict_drag", methods=["POST"])
def predict_drag():
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    img_data = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(img_data)).convert("L").resize((64,64))
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        _, drag_pred = cnn_model(img_tensor)
    return jsonify({"predicted_drag": drag_pred.item()})
@app.route("/")
def index():
    return "API is running. Use /predict_drag for predictions."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
