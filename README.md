# Digital Twin for 2D Unsteady Flow: Drag Prediction

## Overview
This project develops a digital twin for predicting instantaneous drag on a sphere in a 2D unsteady flow. It leverages a two-tier modeling approach:
- **Baseline Model**: A simple MLP on flattened images.
- **Advanced Model**: A CNN for spatial compression combined with an LSTM that evolves latent features over time, augmented with a physics-informed loss (to enforce smooth, physically plausible predictions).
- **Deployment**: Fully containerized with Docker, includes a Flask API for production serving, and features an interactive UI built with Streamlit.

## Repository Structure
project/ ├─ code/ │ ├─ utils/ # Data loading and preprocessing utilities │ │ ├─ csv_utils.py │ │ ├─ image_utils.py │ │ └─ data_loader.py │ ├─ models/ # Baseline, CNN, and LSTM models │ │ ├─ baseline_model.py │ │ ├─ cnn_model.py │ │ └─ lstm_model.py ├─ data/ # Raw data organized by Reynolds number ├─ main.py # Model training script (data splitting, training baseline, CNN, and LSTM with physics-informed loss) ├─ evaluate.py # Model evaluation script ├─ streamlit_app.py # Interactive UI for predictions (includes contour plots) ├─ api.py # Flask API for serving predictions ├─ Dockerfile # Containerization configuration ├─ requirements.txt # Python dependencies └─ README.md # Project documentation and business pitch

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
Train the Models: The training script splits the data into 70% training, 10% validation, and 20% test sets.
python main.py
python evaluate.py
streamlit run streamlit_app.py
python api.py
docker build -t drag-prediction .
docker run -p 8501:8501 drag-prediction

---

# Final Notes

This updated code base now includes:
- A **physics-informed loss** term in the LSTM training to promote smooth (physically plausible) predictions.
- Data splitting into training (70%), validation (10%), and test (20%) sets.
- Comprehensive documentation, Docker containerization, an API, and an interactive UI.


