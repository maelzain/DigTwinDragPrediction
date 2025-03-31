import os
import io
import base64
import logging
import yaml
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

from code.models.cnn_model import CNNDragPredictor
from code.models.baseline_model import BaselineMLP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('api_server.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def load_configuration(config_path="config.yaml"):
    """Load configuration from config.yaml and set minimal defaults."""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("device", "cpu")
        cfg.setdefault("resize", [64, 64])
        return cfg
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        return {"device": "cpu", "resize": [64, 64]}

config = load_configuration()
device = torch.device(config.get("device", "cpu"))

def load_model(model_type="cnn"):
    """Load and return the specified model using parameters from config."""
    try:
        if model_type == "cnn":
            latent_dim = config["cnn"]["latent_dim"]
            model = CNNDragPredictor(latent_dim=latent_dim).to(device)
            model_path = os.path.join(config["model_dir"], "cnn_drag_predictor.pth")
        else:
            input_dim = config["resize"][0] * config["resize"][1]
            hidden_dim = config["baseline"]["hidden_dim"]
            model = BaselineMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
            model_path = os.path.join(config["model_dir"], "baseline_mlp.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"{model_type.upper()} model loaded from {model_path}.")
        return model
    except Exception as e:
        logger.error(f"Model loading failed for {model_type}: {e}")
        raise

# Preload models
cnn_model = load_model("cnn")
mlp_model = load_model("mlp")

def is_valid_contour_image(filename, image):
    """Validate the image using benchmark criteria."""
    expected_keywords = ["timestep", "re", "contour", "velocity"]
    if not any(keyword in filename.lower() for keyword in expected_keywords):
        logger.warning(f"Filename '{filename}' lacks expected keywords.")
        return False
    image_array = np.array(image.convert("L"))
    if np.std(image_array) < 10:
        logger.warning("Insufficient image contrast.")
        return False
    return True

@app.route("/")
def index():
    """Health check endpoint."""
    return jsonify({
        "service": "Digital Twin Drag Prediction API",
        "status": "operational",
        "models": ["CNN", "MLP"]
    }), 200

@app.route("/predict_drag", methods=["POST"])
def predict_drag_endpoint():
    """Predict drag from an uploaded contour image."""
    data = request.get_json(force=True)
    required_fields = ["image", "filename", "model_type", "reynolds_group"]
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        b64_string = data["image"]
        if len(b64_string) % 4:
            b64_string += "=" * (4 - len(b64_string) % 4)
        img_data = base64.b64decode(b64_string)
        resize_dims = tuple(config["resize"])
        image = Image.open(io.BytesIO(img_data)).convert("L").resize(resize_dims)
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return jsonify({"error": "Invalid image data"}), 400

    if not is_valid_contour_image(data["filename"], image):
        logger.error("Uploaded image failed validation.")
        return jsonify({"error": "Invalid contour plot"}), 400

    model_type = data["model_type"].lower()
    reynolds_group = data["reynolds_group"]

    try:
        model = cnn_model if model_type == "cnn" else mlp_model
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            drag_pred = output[1] if isinstance(output, tuple) else output

        # Use drag_ranges from config
        drag_ranges = config.get("drag_ranges", {})
        if reynolds_group in drag_ranges:
            drag_min, drag_max = drag_ranges[reynolds_group]
            predicted_drag = drag_pred.item() * (drag_max - drag_min) + drag_min
        else:
            predicted_drag = drag_pred.item()

        logger.info(f"Predicted drag: {predicted_drag:.8f} using {model_type.upper()} for {reynolds_group}.")
        return jsonify({
            "predicted_drag": predicted_drag,
            "model": model_type,
            "reynolds_group": reynolds_group
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.get("api_port", 5000), debug=False)
