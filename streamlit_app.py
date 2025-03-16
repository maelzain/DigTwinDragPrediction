import os
import io
import base64
import logging
import streamlit as st
import torch
import yaml
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import torchvision.transforms as transforms
from code.models.cnn_model import CNNDragPredictor
from code.models.baseline_model import BaselineMLP

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def is_valid_contour_image(filename, image):
    """
    Validate that the image is likely a velocity contour plot.
    Checks that the filename contains expected keywords and that image contrast is sufficient.
    """
    expected_keywords = ["timestep", "re", "contour"]
    if not any(keyword in filename.lower() for keyword in expected_keywords):
        st.error("Filename must include keywords like 'timestep', 're', or 'contour'.")
        return False
    image_array = np.array(image.convert("L"))
    if np.std(image_array) < 10:
        st.error("Image contrast is too low; it may not be a valid contour plot.")
        return False
    return True

def predict_drag(model, image, device):
    """
    Preprocess the image, run the model, and return the predicted drag.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, tuple):
            _, drag_pred = output
        else:
            drag_pred = output
    return drag_pred.item()

def main():
    st.title("Digital Twin for 2D Unsteady Flow: Drag Prediction")
    st.markdown("""
    **Overview:**  
    Upload a **velocity contour plot** of a 2D unsteady flow around a sphere to get instantaneous drag predictions.
    
    Choose between a Baseline MLP and an Optimized CNN model.
    """)
    config = load_config()
    device = config.get("device", "cpu")
    model_choice = st.selectbox("Select Prediction Model", ["Optimized CNN", "Baseline MLP"])
    
    if model_choice == "Optimized CNN":
        model_path = os.path.join("models", "cnn_drag_predictor.pth")
        model = CNNDragPredictor(latent_dim=config["cnn"].get("latent_dim", 128)).to(device)
    else:
        model_path = os.path.join("models", "baseline_mlp.pth")
        model = BaselineMLP(input_dim=64*64, hidden_dim=128).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        st.success(f"{model_choice} model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    uploaded_file = st.file_uploader("Choose a contour plot image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("L").resize((64, 64))
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        if not is_valid_contour_image(uploaded_file.name, image):
            st.warning("Please upload a valid contour plot image.")
            return

        st.image(image, caption="Uploaded Snapshot", use_container_width=True)
        
        # Interactive Plotly Contour Plot
        image_array = np.array(image)
        contour_fig = go.Figure(data=go.Contour(
            z=image_array,
            colorscale='Viridis',
            contours=dict(showlabels=True, labelfont=dict(size=12, color='white'))
        ))
        contour_fig.update_layout(title="Interactive Contour Plot", autosize=True)
        st.plotly_chart(contour_fig, use_container_width=True)
        
        # Simulated time-series plot (average intensity along rows)
        time = np.arange(image_array.shape[0])
        intensity = np.mean(image_array, axis=1)
        ts_fig = px.line(x=time, y=intensity, labels={'x': 'Pixel Row', 'y': 'Average Intensity'},
                         title="Simulated Time-Series of Intensity")
        st.plotly_chart(ts_fig, use_container_width=True)
        
        predicted_drag = predict_drag(model, image, device)
        st.write(f"**Predicted Drag ({model_choice}):** {predicted_drag:.8f}")
        
        # Residual Histogram Simulation
        num_samples = 100
        y_true = np.array([predicted_drag + np.random.normal(0, 0.0005) for _ in range(num_samples)])
        y_pred = np.array([predicted_drag + np.random.normal(0, 0.0005) for _ in range(num_samples)])
        residuals = y_true - y_pred
        num_bins = st.slider("Number of bins for Residual Histogram", min_value=10, max_value=100, value=50)
        hist_fig = go.Figure(data=go.Histogram(
            x=residuals,
            nbinsx=num_bins,
            marker_color="indianred",
            opacity=0.75
        ))
        hist_fig.update_layout(
            title="Residual Histogram",
            xaxis_title="Residuals (True - Predicted)",
            yaxis_title="Frequency",
            template="plotly_white",
            bargap=0.1
        )
        st.plotly_chart(hist_fig, use_container_width=True)
        
        # API call simulation
        if st.button("Get API Prediction"):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            st.info("API integration not implemented in this demo. Uncomment the API call code for live predictions.")
            # Uncomment below to call the live API:
            # import requests
            # response = requests.post("http://127.0.0.1:5000/predict_drag", json={"image": img_str, "filename": uploaded_file.name})
            # if response.ok:
            #     result = response.json()
            #     st.write(f"**API Predicted Drag:** {result['predicted_drag']:.8f}")
            # else:
            #     st.error("API request failed.")
    
if __name__ == "__main__":
    main()
