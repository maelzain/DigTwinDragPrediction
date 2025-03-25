# Digital Twin for 2D Unsteady Flow Drag Prediction

## 🚀 Project Overview

A cutting-edge machine learning solution integrating traditional Computational Fluid Dynamics (CFD) techniques to predict instantaneous drag forces on a sphere in a 2D unsteady flow field.

### 🌟 Key Features
- **Advanced ML Models:** Baseline MLP and Optimized CNN-LSTM architecture
- **Containerized Deployment:** Docker and Docker Compose support
- **Interactive Interfaces:** Streamlit UI and Flask API
- **Comprehensive Evaluation:** Robust performance metrics and visualization

## 📊 Technical Approach

### Data Preprocessing
- Image Processing: CFD snapshots converted to 64×64 grayscale
- Drag Force Normalization: MinMax scaling for precise prediction
- Data Split: 70% training, 10% validation, 20% testing

### Modeling Pipeline
1. **Baseline Model:** Multilayer Perceptron (MLP)
2. **Optimized Model:** 
   - CNN for feature extraction
   - LSTM for temporal evolution
   - Drag force prediction

## 🛠 Installation & Setup

### Prerequisites
- Git
- Docker
- Docker Compose
- Modern Web Browser

### Quick Start

```bash
# Clone the Repository
git clone https://github.com/maelzain/DigTwinDragPrediction.git
cd DigTwinDragPrediction

# Build Docker Image
docker build -t drag-prediction .

# Launch Services
docker-compose up
```

### Access Services
- **API:** http://localhost:5000
- **Streamlit UI:** http://localhost:8501

## 🧪 Testing & Evaluation

### API Testing
1. Convert image to base64 string
2. Send POST request to `/predict_drag`

### Model Evaluation
```bash
python train.py
python evaluate.py
```

## 🌐 Business Value

Our digital twin tool accelerates engineering design cycles by:
- Reducing simulation time
- Enhancing computational efficiency
- Providing real-time drag predictions
- Supporting aerospace, automotive, and environmental engineering

## 🚧 Project Structure

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
│   └── CFD snapshots by Reynolds number
├── streamlit_app.py
├── api.py
├── Dockerfile
└── docker-compose.yml
```

## 🔭 Future Roadmap
- 3D Flow Field Extension
- Real-Time Sensor Data Integration
- Advanced MLOps Pipelines
- Enhanced Visualization Dashboards

## 🤝 Contributing
Contributions are welcome! Please:
- Open an issue
- Submit a pull request
- Follow project coding standards

## 📞 Contact

**Mahdi ELzain**
- 📧 Email: mahdielzain@outlook.com
- 🐙 GitHub: [maelzain](https://github.com/maelzain/DigTwinDragPrediction)

## 📄 License
AUB

## 💡 Acknowledgments
[DR.AmmarMohanna]