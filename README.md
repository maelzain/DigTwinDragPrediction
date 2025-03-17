# Digital Twin for 2D Unsteady Flow Drag Prediction

Welcome to the **Digital Twin for 2D Unsteady Flow Drag Prediction** project! This repository presents a cutting-edge machine learning solution integrated with traditional CFD techniques to predict instantaneous drag forces on a sphere in a 2D unsteady flow field. Our framework offers both a baseline model and an optimized model, ensuring robust performance, and is fully containerized for seamless deployment via API and interactive UI.

---

## Table of Contents

- [Overview](#overview)
- [Business Vision](#business-vision)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Running the API & UI with Docker Compose](#running-the-api--ui-with-docker-compose)
  - [Testing the API](#testing-the-api)
  - [Training & Evaluation](#training--evaluation)
- [MLOps & Deployment](#mlops--deployment)
- [Future Directions](#future-directions)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Step-by-Step Summary](#step-by-step-summary)

---

## Overview

In modern CFD, achieving both high simulation accuracy and computational efficiency is essential. Our digital twin framework for drag prediction leverages machine learning to:
- **Preprocess Data:** Load and normalize CFD snapshot images (converted to 64×64 grayscale) and corresponding drag force CSV files. Drag forces, which are on the order of 1e-07, are normalized using the MinMax scaler.
- **Modeling Pipeline:**  
  - **Baseline Model:** A simple Multilayer Perceptron (MLP) processes flattened image data to predict drag.
  - **Optimized Model:** A Convolutional Neural Network (CNN) extracts a non-linear latent space from the images, which is further evolved in time using a Long Short-Term Memory (LSTM) network to predict drag forces.
- **Evaluation:** Data is split into 70% training, 10% validation, and 20% testing. Robust cross-validation is performed and regression metrics (MSE, RMSE, R²) are reported alongside evaluation plots.

---

## Business Vision

Our digital twin tool empowers engineers with real-time drag predictions, significantly reducing simulation time and accelerating design cycles across aerospace, automotive, and environmental engineering. By integrating machine learning into traditional CFD workflows, our solution enhances computational efficiency without sacrificing accuracy. Fully cloud-ready and containerized, the product is designed for scalable deployment in modern MLOps pipelines while offering an intuitive user interface and a production-grade API.

---

## Repository Structure

DigitalTwinDragPrediction/ ├─ code/ │ ├─ models/ │ │ ├─ baseline_model.py # Baseline MLP model │ │ ├─ cnn_model.py # CNN for latent feature extraction & drag prediction │ │ └─ lstm_model.py # LSTM for evolving latent features over time │ ├─ utils/ │ │ ├─ csv_utils.py # CSV loading and drag normalization utilities │ │ ├─ image_utils.py # Image loading and preprocessing utilities │ │ └─ data_loader.py # Data loader for Reynolds experiments │ └─ train.py # Training pipeline for CNN and LSTM models ├─ data/ # CFD snapshots and drag force CSV files organized by Reynolds number (e.g., re300) ├─ evaluate.py # Evaluation and visualization script ├─ streamlit_app.py # Interactive UI for drag prediction (Streamlit) ├─ api.py # Flask API for serving model predictions ├─ Dockerfile # Docker configuration for containerization ├─ docker-compose.yml # Docker Compose configuration for API & UI services ├─ requirements.txt # Python dependencies ├─ README.md # Project documentation (this file) └─ .gitignore # Files and directories to ignore


---

## Installation & Setup

### Prerequisites

- **Git:** To clone the repository.
- **Docker & Docker Compose:** For containerization and running the application.
- **A modern web browser**

Verify installations:

```bash
git --version
docker --version
docker-compose --version

#Clone the Repository

git clone https://github.com/maelzain/DigTwinDragPrediction.git
cd DigTwinDragPrediction
#Usage

docker build -t drag-prediction .

docker-compose up
#Testing the API

python -c "import base64; print(base64.b64encode(open(r'C:\Users\user\Desktop\COURSES\AUB\Spring 2024-2025-739988\introduction to machine learning\project 01\GItHub\project\data\re300\timestep_3000.png', 'rb').read()).decode('utf-8'))"
#Start the Services:
docker-compose up

#Access the Services:

API: http://localhost:5000
Streamlit UI: http://localhost:8501
Test the API:

Convert an image to a base64 string.
Send a POST request to /predict_drag using curl.

#Train & Evaluate Models (if needed):
python train.py
python evaluate.py
#The evaluation script splits the data into 70% training, 10% validation, and 20% testing, performs cross-validation, and reports metrics such as MSE, RMSE, and R². Evaluation plots are saved in the evaluation_plots/ directory.


#Stop the Services:

docker-compose down

#MLOps & Deployment
Dockerization: The project is containerized for consistency and reproducibility.
API Serving: The optimized model is served via a production-grade Flask API.
Interactive UI: A user-friendly Streamlit app enables real-time predictions and visualization.
Cloud-Ready: The solution is scalable and can be deployed on cloud platforms such as AWS, GCP, or Azure.
Documentation & GitHub: The repository is organized, version-controlled, and thoroughly documented.
#Future Directions
3D Flow Field Extension: Extend the methodology to three-dimensional flow scenarios.
Real-Time Integration: Incorporate live sensor data for dynamic drag prediction.
Advanced MLOps Pipelines: Implement continuous integration and automated retraining.
Enhanced Visualizations: Develop sophisticated dashboards and interactive visualizations for deeper insights

#Contributing
Contributions are welcome! Please open an issue or submit a pull request with enhancements, bug fixes, or new features.
#Contact
For questions or further information, please contact:

Name: Mahdi ELzain
Email: mahdielzain@outlook.com
GitHub: https://github.com/maelzain/DigTwinDragPrediction