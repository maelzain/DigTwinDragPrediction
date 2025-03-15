import streamlit as st
import torch
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from code.models.cnn_model import CNNDragPredictor

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    st.title("Digital Twin for 2D Unsteady Flow: Drag Prediction")
    st.write("Upload a snapshot image to view its contour plot and get a drag prediction (using the CNN model).")

    config = load_config()
    device = config.get("device", "cpu")
    cnn_model_path = os.path.join("models", "cnn_drag_predictor.pth")

    # Load CNN model for prediction
    cnn_model = CNNDragPredictor(latent_dim=config["cnn"].get("latent_dim", 64)).to(device)
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model.eval()

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Load and process image as grayscale
        image = Image.open(uploaded_file).convert("L").resize((64, 64))
        
        # Convert image to numpy array for contour plot
        image_array = np.array(image)
        
        # Create contour plot
        fig, ax = plt.subplots(figsize=(6, 6))
        contour = ax.contourf(image_array, cmap="viridis")
        ax.set_title("Contour Plot of Snapshot")
        plt.colorbar(contour, ax=ax)
        
        st.pyplot(fig)
        
        # For drag prediction, convert image to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, drag_pred = cnn_model(img_tensor)
        st.write(f"**Predicted Drag (CNN):** {drag_pred.item():.8f}")
        st.write("Note: The contour plot represents the intensity values of the snapshot.")

if __name__ == "__main__":
    main()
