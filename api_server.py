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
    """Load configuration and set defaults."""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("device", "cpu")
        cfg.setdefault("cnn", {"latent_dim": 128})
        return cfg
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        return {"device": "cpu", "cnn": {"latent_dim": 128}}

config = load_configuration()
device = torch.device(config.get("device", "cpu"))

def load_model(model_type="cnn"):
    """Load and return the specified model."""
    try:
        if model_type == "cnn":
            latent_dim = config["cnn"].get("latent_dim", 128)
            model = CNNDragPredictor(latent_dim=latent_dim).to(device)
            model_path = os.path.join("models", "cnn_drag_predictor.pth")
        else:
            model = BaselineMLP(input_dim=64*64, hidden_dim=128).to(device)
            model_path = os.path.join("models", "baseline_mlp.pth")
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
    """Validate the uploaded image based on filename keywords and image contrast."""
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
def predict_drag():
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
        image = Image.open(io.BytesIO(img_data)).convert("L").resize((64, 64))
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return jsonify({"error": "Invalid image data"}), 400

    if not is_valid_contour_image(data["filename"], image):
        logger.error("Uploaded image failed validation.")
        return jsonify({"error": "Invalid contour plot"}), 400

    model_type = data["model_type"].lower()
    reynolds_group = data["reynolds_group"]

    # Drag prediction normalization ranges for different Reynolds groups.
    drag_ranges = {
        "re37": (3.26926e-07, 3.33207e-07),
        "re75": (1.01e-06, 1.04e-06),
        "re150": (3.15e-06, 0.000130279),
        "re300": (1.4e-05, 1.6e-05)
    }

    try:
        model = cnn_model if model_type == "cnn" else mlp_model
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            drag_pred = output[1] if isinstance(output, tuple) else output

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
    app.run(host="0.0.0.0", port=5000, debug=False)
