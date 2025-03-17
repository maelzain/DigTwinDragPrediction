# Digital Twin for 2D Unsteady Flow Drag Prediction

Welcome to the **Digital Twin for 2D Unsteady Flow Drag Prediction** project! This repository presents a state-of-the-art machine learning solution integrated with traditional CFD techniques to predict instantaneous drag forces on a sphere in a 2D unsteady flow field. Our approach includes both a baseline model and an optimized model, ensuring robust performance with comprehensive evaluation and interactive visualizations.

---

## Table of Contents

- [Overview](#overview)
- [Business Vision](#business-vision)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Interactive UI (Streamlit)](#interactive-ui-streamlit)
  - [API (Flask)](#api-flask)
- [Training & Evaluation](#training--evaluation)
- [MLOps & Deployment](#mlops--deployment)
- [Future Directions](#future-directions)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

In modern computational fluid dynamics (CFD), achieving a balance between simulation accuracy and computational efficiency is paramount. Our project develops a **digital twin** for predicting the instantaneous drag force on a sphere. The pipeline includes:

1. **Data Preprocessing:**  
   - Loading images from CFD experiments and corresponding drag force CSV files.
   - Resizing and optionally augmenting snapshots (64×64 grayscale images).
   - Normalizing very low drag force values using the MinMax scaler (ideal for values on the order of 1e-07).

2. **Modeling Pipeline:**  
   - **Baseline Model:** A simple Multilayer Perceptron (MLP) that processes flattened snapshots.
   - **Optimized Model:** A Convolutional Neural Network (CNN) that extracts latent features from images, followed by a Long Short-Term Memory (LSTM) network that evolves these latent features over time to predict drag forces.

3. **Robust Evaluation:**  
   - Cross-validation using a split of 70% training, 10% validation, and 20% testing.
   - Reporting regression metrics (MSE, RMSE, R²) and generating insightful evaluation plots.

---

## Business Vision

Imagine a tool that enables engineers to rapidly predict drag forces in real time, thus reducing simulation time and accelerating design iterations across aerospace, automotive, and environmental engineering. Our digital twin is cloud-ready, seamlessly integrates into MLOps pipelines, and offers an interactive UI and production-grade API. This solution not only enhances computational efficiency but also provides a competitive edge by enabling faster and more accurate simulations.

---

## Repository Structure

DigitalTwinDragPrediction/ ├─ code/ │ ├─ models/ │ │ ├─ baseline_model.py # Baseline MLP model │ │ ├─ cnn_model.py # CNN for latent feature extraction & drag prediction │ │ └─ lstm_model.py # LSTM for evolving latent features over time │ ├─ utils/ │ │ ├─ csv_utils.py # CSV loading and normalization utilities │ │ ├─ image_utils.py # Image loading and preprocessing utilities │ │ └─ data_loader.py # Data loader for Reynolds experiments │ └─ train.py # Training pipeline for CNN and LSTM models ├─ data/ # Data organized by Reynolds number (e.g., re300) ├─ evaluate.py # Evaluation and visualization script ├─ streamlit_app.py # Interactive UI for drag prediction (Streamlit) ├─ api.py # Flask API for model serving ├─ Dockerfile # Docker configuration for containerization ├─ docker-compose.yml # Docker Compose configuration for API & UI ├─ requirements.txt # Python dependencies ├─ README.md # Project documentation (this file) └─ .gitignore # Files and directories to ignore

yaml
Copy

---

## Installation & Setup

### Prerequisites

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
- A modern web browser

### Clone the Repository

```bash
git clone https://github.com/maelzain/DigTwinDragPrediction.git
cd DigTwinDragPrediction
Build the Docker Image
bash
Copy
docker build -t drag-prediction .
Run the Services Using Docker Compose
bash
Copy
docker-compose up
This will launch:

Flask API on port 5000
Streamlit UI on port 8501
Note: If you experience port conflicts, ensure that no other processes are using these ports.

Usage
Interactive UI (Streamlit)
Open your browser and navigate to http://localhost:8501. Here you can:

Upload a velocity contour plot image (ensure the filename contains keywords like "timestep", "re", or "contour").
View interactive visualizations, including contour plots and simulated time-series.
Get real-time drag predictions from both the baseline and optimized models.
API (Flask)
Access the API at http://localhost:5000.
To test the API, send a POST request to /predict_drag with a JSON payload:

bash
Copy
curl -X POST http://localhost:5000/predict_drag \
    -H "Content-Type: application/json" \
    -d '{"image": "<base64_string_here>", "filename": "timestep_940.png"}'
Replace <base64_string_here> with the base64-encoded string of your image. Use a command such as:

bash
Copy
python -c "import base64; print(base64.b64encode(open(r'C:\path\to\timestep_3000.png', 'rb').read()).decode('utf-8'))"
Training & Evaluation
Data Preprocessing
Images: Loaded from directories (e.g., data/re300) are converted to grayscale, resized to 64×64 pixels, and optionally augmented.
Drag Forces: Loaded from CSV files, then normalized using the MinMax scaler—ideal for low magnitude values (~1e-07).
Model Training
Baseline Model (MLP): Processes flattened image data to predict drag.
Optimized Model (CNN+LSTM): A CNN extracts latent features from images; an LSTM evolves these features to predict drag across time.
Robust Evaluation
We use cross-validation (70% training, 10% validation, 20% testing) to evaluate model performance:

Metrics reported: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R².
Evaluation plots are generated to compare baseline and optimized models.
Run evaluation with:

bash
Copy
python evaluate.py
MLOps & Deployment
Dockerization: All components are containerized for consistency and reproducibility.
API Serving: Models are served via a Flask API.
Interactive UI: A Streamlit app provides a user-friendly interface for real-time predictions.
GitHub & Documentation: The repository is structured for clarity, version control, and collaboration.
Cloud-Ready: The solution is scalable and ready for deployment on platforms such as AWS, GCP, or Azure.
Future Directions
3D Flow Field Extension: Extend the methodology to three-dimensional flow scenarios.
Real-Time Integration: Incorporate real-time sensor data for dynamic drag prediction.
Advanced MLOps Pipelines: Implement continuous integration and automated retraining pipelines.
Enhanced Visualizations: Develop more sophisticated dashboards and interactive visualizations.
Contributing
Contributions are welcome! Please open an issue or submit a pull request with enhancements or bug fixes.

License
This project is licensed under the MIT License.

Contact
For questions or further information, please contact:

Name: [Mahdi ELzain]
Email: [mahdielzain@outlook.com]
GitHub: https://github.com/maelzain/DigTwinDragPrediction
yaml
Copy

---

### How to Use This README

1. Copy the entire content above.
2. Create or update the `README.md` file in your repository's root directory.
3. Paste the content into the file and save.
4. Commit and push your changes to GitHub.

