import os
import io
import base64
import logging
import yaml
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageStat
import numpy as np

from code.models.cnn_model import CNNDragPredictor
from code.models.baseline_model import BaselineMLP

# Configure logging with more comprehensive output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load configuration with error handling
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    config = {"device": "cpu"}

device = torch.device(config.get("device", "cpu"))

def load_model(model_type="cnn"):
    """
    Load and configure ML model for drag prediction
    
    Args:
        model_type (str): Type of model to load ('cnn' or 'mlp')
    
    Returns:
        torch.nn.Module: Loaded and configured model
    """
    try:
        if model_type == "cnn":
            model = CNNDragPredictor(latent_dim=config["cnn"].get("latent_dim", 128)).to(device)
            model_path = os.path.join("models", "cnn_drag_predictor.pth")
        else:
            model = BaselineMLP(input_dim=64*64, hidden_dim=128).to(device)
            model_path = os.path.join("models", "baseline_mlp.pth")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"{model_type.upper()} model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

# Preload models
cnn_model = load_model("cnn")
mlp_model = load_model("mlp")

def is_valid_contour_image(filename, image):
    """
    Validate velocity contour plot with enhanced checks
    
    Args:
        filename (str): Name of uploaded image file
        image (PIL.Image): Loaded image
    
    Returns:
        bool: Image validity status
    """
    expected_keywords = ["timestep", "re", "contour", "velocity"]
    if not any(keyword in filename.lower() for keyword in expected_keywords):
        logger.warning(f"Filename {filename} lacks expected keywords.")
        return False
    
    # More robust contrast check using NumPy
    image_array = np.array(image.convert("L"))
    contrast_threshold = 10
    if np.std(image_array) < contrast_threshold:
        logger.warning("Insufficient image contrast.")
        return False
    
    return True

@app.route("/")
def index():
    """Health check endpoint"""
    return jsonify({
        "service": "Digital Twin Drag Prediction API",
        "status": "operational",
        "models": ["CNN", "MLP"]
    }), 200

@app.route("/predict_drag", methods=["POST"])
def predict_drag():
    """
    Prediction endpoint for drag calculation
    
    Expects JSON with base64 encoded image and filename
    Returns predicted drag value
    """
    data = request.get_json(force=True)
    required_fields = ["image", "filename", "model_type", "reynolds_group"]
    
    # Validate input
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        b64_string = data["image"]
        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += "=" * (4 - missing_padding)
        
        img_data = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(img_data)).convert("L").resize((64, 64))
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return jsonify({"error": "Invalid image data"}), 400

    filename = data["filename"]
    if not is_valid_contour_image(filename, image):
        return jsonify({"error": "Invalid contour plot"}), 400

    model_type = data["model_type"].lower()
    reynolds_group = data["reynolds_group"]
    
    # Drag ranges for normalization
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

        # Denormalize prediction to physical units
        if reynolds_group in drag_ranges:
            drag_min, drag_max = drag_ranges[reynolds_group]
            predicted_drag = drag_pred.item() * (drag_max - drag_min) + drag_min
        else:
            predicted_drag = drag_pred.item()

        logger.info(f"Drag prediction: {predicted_drag:.8f} for {model_type} model")
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