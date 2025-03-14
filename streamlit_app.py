import streamlit as st
import torch
import os
import yaml
import numpy as np
from PIL import Image

from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    st.title("Digital Twin for 2D Unsteady Flow: Drag Prediction")

    # Load config
    config = load_config()
    device = config.get("device", "cpu")

    # Load models
    cnn_model_path = os.path.join("models", "cnn_drag_predictor.pth")
    lstm_model_path = os.path.join("models", "lstm_drag.pth")

    cnn_model = CNNDragPredictor(latent_dim=config["cnn"].get("latent_dim", 64)).to(device)
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model.eval()

    lstm_model = LSTMDragPredictor(
        input_size=config["lstm"].get("input_size", 64),
        hidden_size=config["lstm"].get("hidden_size", 32),
        num_layers=config["lstm"].get("num_layers", 1),
        output_size=config["lstm"].get("output_size", 1)
    ).to(device)
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    lstm_model.eval()

    st.write("Upload a single snapshot (PNG/JPG) for quick drag prediction (CNN only).")

    uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L").resize((64,64))
        st.image(image, caption="Uploaded Snapshot", use_column_width=True)
        # Convert to tensor
        import torchvision.transforms as transforms
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(device)  # (1, 1, 64, 64)

        with torch.no_grad():
            latent, drag_pred = cnn_model(img_tensor)
            st.write(f"**CNN Predicted Drag (single snapshot):** {drag_pred.item():.6f}")

        # For LSTM, you need a sequence of latents. This UI only handles one image. 
        # In a real scenario, you'd gather multiple snapshots to form a time series.

if __name__ == "__main__":
    main()
