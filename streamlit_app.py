import os
import time
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
    """Validate image using benchmark criteria."""
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
    """Predict drag using a minimal transform (only ToTensor) as in the benchmark."""
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
    st.title("Instant Drag Prediction for 2D Unsteady Flow")
    st.markdown("""
    **Overview:**  
    Upload a **velocity contour plot** of a 2D unsteady flow around a sphere to instantly receive a drag prediction.
    
    **Why It Matters:**  
    Traditional CFD simulations can take around 1 hour with significant computational cost.
    Our solution delivers predictions in seconds—saving you time and money.
    """)
    
    config = load_config()
    device = torch.device(config.get("device", "cpu"))
    model_choice = st.selectbox("Select Prediction Model", ["Optimized CNN", "Baseline MLP"], key="model_choice")
    reynolds_group = st.selectbox("Select Reynolds Group", ["re37", "re75", "re150", "re300"], key="reynolds_group")
    
    if model_choice == "Optimized CNN":
        model_path = os.path.join(config["model_dir"], "cnn_drag_predictor.pth")
        latent_dim = config["cnn"]["latent_dim"]
        model = CNNDragPredictor(latent_dim=latent_dim).to(device)
    else:
        model_path = os.path.join(config["model_dir"], "baseline_mlp.pth")
        input_dim = config["resize"][0] * config["resize"][1]
        hidden_dim = config["baseline"]["hidden_dim"]
        model = BaselineMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        st.success(f"{model_choice} model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    uploaded_file = st.file_uploader("Upload a contour plot image...", type=["png", "jpg", "jpeg"], key="uploaded_file")
    if uploaded_file is not None:
        try:
            resize_dims = tuple(config["resize"])
            image = Image.open(uploaded_file).convert("L").resize(resize_dims)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        if not is_valid_contour_image(uploaded_file.name, image):
            st.warning("Please upload a valid contour plot image.")
            return

        st.image(image, caption="Uploaded Snapshot", use_container_width=True)

        image_array = np.array(image)
        contour_fig = go.Figure(data=go.Contour(
            z=image_array,
            colorscale='Viridis',
            contours=dict(showlabels=True, labelfont=dict(size=12, color='white'))
        ))
        contour_fig.update_layout(title="Interactive Contour Plot", autosize=True)
        st.plotly_chart(contour_fig, use_container_width=True)
        
        time_axis = np.arange(image_array.shape[0])
        intensity = np.mean(image_array, axis=1)
        ts_fig = px.line(x=time_axis, y=intensity,
                         labels={'x': 'Pixel Row', 'y': 'Average Intensity'},
                         title="Simulated Time-Series of Intensity")
        st.plotly_chart(ts_fig, use_container_width=True)
        
        start_time = time.time()
        drag_norm = predict_drag(model, image, device)
        inference_time = time.time() - start_time
        
        # Reverse normalization calculation using drag_ranges from config
        drag_ranges = config.get("drag_ranges", {})
        if reynolds_group in drag_ranges:
            drag_min, drag_max = drag_ranges[reynolds_group]
            predicted_drag = drag_norm * (drag_max - drag_min) + drag_min
        else:
            predicted_drag = drag_norm
        
        st.markdown(f"**Predicted Drag ({model_choice}):** {predicted_drag:.8f} (physical units)")
        st.markdown(f"**Inference Time:** {inference_time:.3f} seconds")
        
        typical_simulation_time = 3600  # seconds (example)
        typical_simulation_cost = 50.0    # dollars per simulation (example)
        time_saved = typical_simulation_time - inference_time
        percent_time_saved = (time_saved / typical_simulation_time) * 100
        
        st.markdown(f"**Time Saved:** {time_saved:.1f} seconds ({percent_time_saved:.2f}% reduction compared to a traditional CFD simulation)")
        st.markdown(f"**Estimated Cost Savings:** Approximately ${typical_simulation_cost:.2f} saved per simulation by using the ML model.")
        st.markdown("For more details on simulation performance and cost benchmarks, refer to the [SciTASIC Cluster Report](https://www.epfl.ch/campus/services/wp-content/uploads/2019/05/SCITASIC-CLUSTER_2018.pdf).")
        
        st.markdown("---")
        st.markdown("### Additional Evaluation Visualizations")
        eval_plot_path = os.path.join("evaluation_plots", "CNN+LSTM_test_vs_true.png")
        if os.path.exists(eval_plot_path):
            with open(eval_plot_path, "rb") as file:
                st.download_button(
                    label="Download Final Evaluation Plot",
                    data=file,
                    file_name="cnn_lstm_final_comparison.png",
                    mime="image/png"
                )
        else:
            st.info("Evaluation plot not available. Run the evaluation script to generate plots.")
            
if __name__ == "__main__":
    main()
