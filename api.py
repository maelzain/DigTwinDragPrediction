import os
import io
import base64
import logging
import yaml
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image, ImageStat
from code.models.cnn_model import CNNDragPredictor

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
device = config.get("device", "cpu")

# Initialize and load the optimized CNN model
cnn_model = CNNDragPredictor(latent_dim=config["cnn"].get("latent_dim", 128)).to(device)
model_path = os.path.join("models", "cnn_drag_predictor.pth")
cnn_model.load_state_dict(torch.load(model_path, map_location=device))
cnn_model.eval()
logging.info("Optimized CNN model loaded and ready.")

def is_valid_contour_image(filename, image):
    """
    Validate that the image is likely a velocity contour plot.
    Checks that the filename contains expected keywords and the image has sufficient contrast.
    """
    expected_keywords = ["timestep", "re", "contour"]
    if not any(keyword in filename.lower() for keyword in expected_keywords):
        logging.warning("Filename does not contain expected keywords.")
        return False
    # Compute image statistics to assess contrast
    stat = ImageStat.Stat(image)
    if stat.stddev[0] < 10:
        logging.warning("Image contrast is too low; not a valid contour plot.")
        return False
    return True

@app.route("/")
def index():
    return "Digital Twin API for 2D Unsteady Flow Drag Prediction is running."

@app.route("/predict_drag", methods=["POST"])
def predict_drag():
    """
    Expects a JSON payload with:
      - 'image': a base64-encoded image string.
      - 'filename': name of the image file.
    Returns a JSON with the predicted drag value.
    """
    data = request.get_json(force=True)
    if "image" not in data or "filename" not in data:
        return jsonify({"error": "Both 'image' and 'filename' must be provided."}), 400

    try:
        b64_string = data["image"]
        # Ensure proper padding for base64 string
        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += "=" * (4 - missing_padding)
        img_data = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(img_data)).convert("L").resize((64, 64))
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

    filename = data["filename"]
    if not is_valid_contour_image(filename, image):
        return jsonify({"error": "Uploaded image does not appear to be a valid contour plot."}), 400

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        _, drag_pred = cnn_model(img_tensor)
    predicted_drag = drag_pred.item()
    logging.info(f"Predicted drag: {predicted_drag:.8f}")
    return jsonify({"predicted_drag": predicted_drag})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
