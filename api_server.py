#!/usr/bin/env python
"""
API Server for Instant Design Insights

This Flask-based server loads pre-trained models for predicting aerodynamic
drag based on uploaded design snapshot images. It provides a single endpoint
that accepts a JSON payload, validates and processes the image, and returns
the predicted performance metric in physical units.
"""

import os
import io
import base64
import logging
import yaml
import torch
import torchvision.transforms as transforms
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

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
    """
    Load configuration from a YAML file.
    Minimal defaults are set if loading fails.
    """
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("device", "cpu")
        cfg.setdefault("resize", [64, 64])
        return cfg
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {"device": "cpu", "resize": [64, 64]}


config = load_configuration()
device = torch.device(config.get("device", "cpu"))


def load_model(model_type="cnn"):
    """
    Load and initialize the desired model.

    Args:
        model_type (str): 'cnn' for advanced model or 'mlp' for baseline model.

    Returns:
        model (torch.nn.Module): The loaded and evaluated model.
    """
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
        logger.info(f"{model_type.upper()} model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        logger.error(f"Error loading {model_type} model: {e}")
        raise


# Pre-load the models.
advanced_model = load_model("cnn")
standard_model = load_model("mlp")


def is_valid_simulation_image(filename, image):
    """
    Validate the snapshot image by checking its filename and contrast.

    Args:
        filename (str): The name of the uploaded file.
        image (PIL.Image): The image in PIL format.

    Returns:
        bool: True if the image meets the criteria; False otherwise.
    """
    expected_keywords = ["snapshot", "design", "performance", "timestep"]
    if not any(keyword in filename.lower() for keyword in expected_keywords):
        logger.warning(f"Filename '{filename}' does not contain expected keywords.")
        return False
    image_array = np.array(image.convert("L"))
    if np.std(image_array) < 5:
        logger.warning("Image contrast is too low. Snapshot might be invalid.")
        return False
    return True


@app.route("/")
def index():
    """Health-check endpoint."""
    return jsonify({
        "service": "Instant Design Insight API",
        "status": "operational",
        "models": ["Advanced Model", "Standard Model"]
    }), 200


@app.route("/predict_performance", methods=["POST"])
def predict_performance_endpoint():
    """
    Endpoint for predicting aerodynamic drag performance.

    Expects a JSON payload with keys:
      - image: Base64 encoded image string.
      - filename: Name of the image file.
      - model_type: Either "Advanced Model" (cnn) or "Standard Model" (mlp).
      - reynolds_group: Design category identifier.

    Returns:
      JSON response with predicted performance metric.
    """
    data = request.get_json(force=True)
    required_fields = ["image", "filename", "model_type", "reynolds_group"]
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        b64_string = data["image"]
        # Ensure proper padding on the base64 string.
        if len(b64_string) % 4 != 0:
            b64_string += "=" * (4 - len(b64_string) % 4)
        img_data = base64.b64decode(b64_string)
        resize_dims = tuple(config["resize"])
        image = Image.open(io.BytesIO(img_data)).convert("L").resize(resize_dims)
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return jsonify({"error": "Invalid image data"}), 400

    if not is_valid_simulation_image(data["filename"], image):
        logger.error("Image validation failed.")
        return jsonify({"error": "Invalid design snapshot"}), 400

    model_type = data["model_type"].lower()
    reynolds_group = data["reynolds_group"]
    try:
        model = advanced_model if model_type == "cnn" else standard_model
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            performance_pred = output[1] if isinstance(output, tuple) else output

        performance_ranges = config.get("drag_ranges", {})
        if reynolds_group in performance_ranges:
            range_min, range_max = performance_ranges[reynolds_group]
            predicted_performance = performance_pred.item() * (range_max - range_min) + range_min
        else:
            predicted_performance = performance_pred.item()

        response_value = f"{predicted_performance:.8f} (physical units)"
        logger.info(f"Prediction: {response_value} using {model_type.upper()} model for group {reynolds_group}")

        investor_mapping = {
            "re37": "Category A",
            "re75": "Category B",
            "re150": "Category C",
            "re300": "Category D"
        }
        friendly_category = investor_mapping.get(reynolds_group, reynolds_group)

        return jsonify({
            "predicted_performance": response_value,
            "model": model_type,
            "design_category": friendly_category
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    port = config.get("api_port", 5000)
    app.run(host="0.0.0.0", port=port, debug=False)
