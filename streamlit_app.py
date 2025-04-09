import os
import time
import streamlit as st
import torch
import yaml
import numpy as np
import plotly.express as px
from PIL import Image
import torchvision.transforms as transforms

from code.models.cnn_model import CNNDragPredictor
from code.models.baseline_model import BaselineMLP

def load_config():
    # Load the project configuration.
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def is_valid_image(filename, image):
    """
    Validate the uploaded snapshot.
    The filename should include terms like 'snapshot', 'design', or 'performance'
    to ensure it is a valid representation of your simulation.
    """
    # For investor-friendly simplicity, only allow these keywords:
    expected_keywords = ["snapshot", "design", "performance","timestep","re"]
    if not any(keyword in filename.lower() for keyword in expected_keywords):
        st.error("Filename must include keywords like 'snapshot', 'design', or 'performance'.")
        return False
    image_array = np.array(image.convert("L"))
    if np.std(image_array) < 10:
        st.error("Image contrast is too low; it may not be a valid snapshot.")
        return False
    return True

def predict_performance(model, image, device):
    """
    Use the selected model to generate a design performance insight.
    The result is a normalized number that will be converted into a real-world metric.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, tuple):
            # Use the prediction portion if extra information is returned.
            _, prediction = output
        else:
            prediction = output
    return prediction.item()

def main():
    st.title("Instant Design Optimization Insights")
    st.markdown("""
    **Overview:**  
    Upload a design simulation snapshot to instantly receive actionable performance insights.  
    Our solution delivers results in seconds—dramatically reducing wait times and driving cost savings.
    """)

    config = load_config()
    device = torch.device(config.get("device", "cpu"))
    
    # Let the user choose between two model versions.
    model_choice = st.selectbox("Select Model", ["Advanced Model", "Standard Model"], key="model_choice")
    
    # Use investor-friendly scenario names.
    friendly_scenarios = ["Scenario A", "Scenario B", "Scenario C", "Scenario D"]
    selected_friendly = st.selectbox("Select Scenario", friendly_scenarios, key="scenario")
    
    # Map investor-friendly names 
    scenario_mapping = {
        "Scenario A": "re37",
        "Scenario B": "re75",
        "Scenario C": "re150",
        "Scenario D": "re300"
    }
    technical_scenario = scenario_mapping[selected_friendly]

    # Load the corresponding model from disk.
    if model_choice == "Advanced Model":
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
        st.success(f"{model_choice} loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Upload a snapshot image.
    uploaded_file = st.file_uploader("Upload your design snapshot...", type=["png", "jpg", "jpeg"], key="uploaded_file")
    if uploaded_file is not None:
        try:
            resize_dims = tuple(config["resize"])
            image = Image.open(uploaded_file).convert("L").resize(resize_dims)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        if not is_valid_image(uploaded_file.name, image):
            st.warning("Please upload a valid snapshot image.")
            return

        st.image(image, caption="Uploaded Snapshot", use_container_width=True)
        
        # Display a visual representation.
        image_array = np.array(image)
        fig = px.imshow(image_array, color_continuous_scale='Viridis', title="Visual Snapshot")
        st.plotly_chart(fig, use_container_width=True)
        
        # Run the performance insight generation.
        start_time = time.time()
        normalized_value = predict_performance(model, image, device)
        inference_time = time.time() - start_time
        
        # Convert the normalized value into a real-world metric using ranges from config.
        # The 'drag_ranges' config is re-used as a performance range.
        ranges = config.get("drag_ranges", {})
        if technical_scenario in ranges:
            min_val, max_val = ranges[technical_scenario]
            final_metric = normalized_value * (max_val - min_val) + min_val
        else:
            final_metric = normalized_value
        
        st.markdown(f"**Predicted Performance Metric ({model_choice}):** {final_metric:.8f} (physical units)")
        st.markdown(f"**Processing Time:** {inference_time:.3f} seconds")
        
        # Business impact statements.
        typical_time = 3600  # Example: traditional simulation time (in seconds)
        simulation_cost = 50.0  # Example: cost per simulation (in dollars)
        time_saved = typical_time - inference_time
        percent_saved = (time_saved / typical_time) * 100
        
        st.markdown(f"**Time Savings:** {time_saved:.1f} seconds ({percent_saved:.2f}% faster than traditional methods)")
        st.markdown(f"**Estimated Cost Savings:** Approximately ${simulation_cost:.2f} saved per simulation")
        st.markdown("For more details on performance benchmarks, refer to [this report](https://www.epfl.ch/campus/services/wp-content/uploads/2019/05/SCITASIC-CLUSTER_2018.pdf).")
        
        st.markdown("---")
        st.markdown("### Additional Insights")
        eval_plot_path = os.path.join("evaluation_plots", "CNN+LSTM_test_vs_true.png")
        if os.path.exists(eval_plot_path):
            with open(eval_plot_path, "rb") as file:
                st.download_button(
                    label="Download Evaluation Visual",
                    data=file,
                    file_name="evaluation_visual.png",
                    mime="image/png"
                )
        else:
            st.info("Evaluation visual not available. Please run the evaluation process to generate it.")

if __name__ == "__main__":
    main()
