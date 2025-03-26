# Digital Twin for 2D Unsteady Flow Drag Prediction

## Overview

This project presents a state-of-the-art machine learning framework that integrates traditional Computational Fluid Dynamics (CFD) techniques with modern deep learning methods to predict instantaneous drag forces on a sphere in a 2D unsteady flow field. The solution is designed to significantly reduce simulation time and computational cost, enabling real-time analysis and design optimization.

## 🚀 Key Features

- **Advanced ML Models:** Deploys both a baseline Multilayer Perceptron (MLP) and an optimized CNN-LSTM architecture for robust drag prediction
- **Containerized Deployment:** Fully containerized with Docker and Docker Compose, ensuring consistency and scalability across environments
- **Interactive Interfaces:** Features an intuitive Streamlit UI for real-time user interaction and a Flask-based API for programmatic access
- **Comprehensive Evaluation:** Provides detailed performance metrics and visualizations to validate and benchmark model predictions against traditional CFD simulations

## 🛠 Technical Approach

### Data Preprocessing

- **Image Processing:** CFD snapshots are converted to 64×64 grayscale images
- **Normalization:** Drag force values are normalized using MinMax scaling
- **Data Splitting:** Dataset divided into 70% training, 10% validation, and 20% testing sets

### Modeling Pipeline

1. **Baseline Model:** Multilayer Perceptron (MLP) for initial drag force predictions
2. **Optimized Model:**
   - **CNN:** Extracts spatial features from CFD snapshots
   - **LSTM:** Captures temporal evolution of latent states
   - **Drag Prediction:** Combines CNN-extracted features and LSTM temporal dynamics

## 🔧 Installation & Setup

### Prerequisites

- Git
- Docker & Docker Compose
- Modern Web Browser

### Quick Start

#### Option A: Deploy Using Docker

```bash
git clone https://github.com/maelzain/DigTwinDragPrediction.git
cd DigTwinDragPrediction
docker-compose up --build
```

Access services:
- **Streamlit UI:** http://localhost:8501
- **API:** http://localhost:5000

#### Option B: Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 🧪 Testing & Evaluation

### API Testing

- Convert contour plot image to base64 string
- Send POST request to `/predict_drag` endpoint

### Model Evaluation

```bash
python train.py
python evaluate.py
```

## 💼 Business Value

- Accelerates engineering design cycles
- Enhances computational efficiency
- Provides real-time drag force estimates for aerospace, automotive, and environmental engineering

## 📂 Project Structure

```
DigitalTwinDragPrediction/
├── code/
│   ├── models/
│   │   ├── baseline_model.py
│   │   ├── cnn_model.py
│   │   └── lstm_model.py
│   ├── utils/
│   │   ├── csv_utils.py
│   │   ├── image_utils.py
│   │   └── data_loader.py
│   └── train.py
├── data/
│   └── CFD snapshots (organized by Reynolds number)
├── streamlit_app.py
├── api_server.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚧 Future Roadmap

- Extend framework to 3D flow fields
- Integrate real-time sensor data
- Develop advanced MLOps pipelines
- Enhance visualization dashboards

## 🤝 Contributing

Contributions are welcome! To contribute:
- Open an issue for discussion
- Submit a pull request
- Follow established coding standards

## 📞 Contact

**Mahdi ELzain**
- **Email:** mahdielzain@outlook.com
- **GitHub:** [@maelzain](https://github.com/maelzain)

## 📄 License

AUB

## 🙏 Acknowledgments

Special thanks to Dr. Ammar Mohanna for valuable guidance and support.