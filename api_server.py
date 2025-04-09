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

# Configure logging in plain language
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('api_server.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def load_configuration(config_path="config.yaml"):
    """Load configuration from config.yaml and set minimal defaults for investor insights."""
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
    """Load and return the chosen performance model (advanced or standard) using configuration settings."""
    try:
        if model_type == "cnn":
            latent_dim = config["cnn"]["latent_dim"]
            # Using our advanced performance model to capture design insights.
            model = CNNDragPredictor(latent_dim=latent_dim).to(device)
            model_path = os.path.join(config["model_dir"], "cnn_drag_predictor.pth")
        else:
            input_dim = config["resize"][0] * config["resize"][1]
            hidden_dim = config["baseline"]["hidden_dim"]
            # Standard model for generating design insights.
            model = BaselineMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
            model_path = os.path.join(config["model_dir"], "baseline_mlp.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"{model_type.upper()} performance model loaded from {model_path}.")
        return model
    except Exception as e:
        logger.error(f"Performance model loading failed for {model_type}: {e}")
        raise

# Preload models
advanced_model = load_model("cnn")
standard_model = load_model("mlp")

def is_valid_simulation_image(filename, image):
    """
    Validate the image using basic criteria to ensure it represents a design snapshot.
    Now accepts filenames containing 'snapshot', 'design', 'performance', or 'timestep',
    and requires a minimum contrast threshold of 5.
    """
    expected_keywords = ["snapshot", "design", "performance", "timestep"]
    if not any(keyword in filename.lower() for keyword in expected_keywords):
        logger.warning(f"Filename '{filename}' lacks expected keywords.")
        return False
    image_array = np.array(image.convert("L"))
    # Relaxed threshold: require standard deviation >= 5.
    if np.std(image_array) < 5:
        logger.warning("Insufficient image contrast – the snapshot may not be valid.")
        return False
    return True

@app.route("/")
def index():
    """Simple health check endpoint."""
    return jsonify({
        "service": "Instant Design Insight API",
        "status": "operational",
        "models": ["Advanced Model", "Standard Model"]
    }), 200

@app.route("/predict_performance", methods=["POST"])
def predict_performance_endpoint():
    """
    Generate a performance insight from an uploaded design snapshot.
    The API expects an image, filename, model type selection, and a design category.
    """
    data = request.get_json(force=True)
    required_fields = ["image", "filename", "model_type", "reynolds_group"]
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        # Decode the uploaded image (in base64).
        b64_string = data["image"]
        if len(b64_string) % 4:
            b64_string += "=" * (4 - len(b64_string) % 4)
        img_data = base64.b64decode(b64_string)
        resize_dims = tuple(config["resize"])
        # Convert image to a format suitable for our model.
        image = Image.open(io.BytesIO(img_data)).convert("L").resize(resize_dims)
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return jsonify({"error": "Invalid image data"}), 400

    if not is_valid_simulation_image(data["filename"], image):
        logger.error("Uploaded image failed validation.")
        return jsonify({"error": "Invalid design snapshot"}), 400

    model_type = data["model_type"].lower()
    design_category = data["reynolds_group"]

    try:
        # Select the appropriate model.
        model = advanced_model if model_type == "cnn" else standard_model
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            performance_pred = output[1] if isinstance(output, tuple) else output

        # Convert the normalized output into a real-world performance metric.
        performance_ranges = config.get("drag_ranges", {})
        if design_category in performance_ranges:
            perf_min, perf_max = performance_ranges[design_category]
            predicted_performance = performance_pred.item() * (perf_max - perf_min) + perf_min
        else:
            predicted_performance = performance_pred.item()

        # Format the final output with the physical unit appended.
        response_value = f"{predicted_performance:.8f} (physical units)"
        logger.info(f"Predicted performance: {response_value} using {model_type.upper()} model for category {design_category}.")
        
        # Map technical design category to investor-friendly label.
        investor_mapping = {
            "re37": "Category A",
            "re75": "Category B",
            "re150": "Category C",
            "re300": "Category D"
        }
        friendly_category = investor_mapping.get(design_category, design_category)
        
        return jsonify({
            "predicted_performance": response_value,
            "model": model_type,
            "design_category": friendly_category
        })
    except Exception as e:
        logger.error(f"Performance prediction error: {e}")
        return jsonify({"error": "Performance insight prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.get("api_port", 5000), debug=False)
