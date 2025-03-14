# Digital Twin for 2D Unsteady Flow: Drag Prediction

## Overview
This project develops a digital twin for predicting instantaneous drag on a sphere in 2D unsteady flow. It includes:
- **Baseline Model**: A simple MLP on flattened images.
- **Advanced Model**: A CNN for spatial compression combined with an LSTM for temporal evolution.
- **Deployment**: Docker containerization, Flask API, and a Streamlit interactive UI.

## Repository Structure
project/ ├─ code/ │ ├─ utils/ # Data loading and preprocessing utilities │ ├─ models/ # Baseline, CNN, and LSTM models ├─ data/ # Raw data organized by Reynolds number ├─ main.py # Model training ├─ evaluate.py # Model evaluation ├─ streamlit_app.py # Interactive UI for predictions ├─ api.py # Flask API for serving predictions ├─ Dockerfile # Containerization ├─ requirements.txt # Dependencies └─ README.md # Project documentation and business pitch


## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python main.py
    python evaluate.py

streamlit run streamlit_app.py
python api.py
docker build -t drag-prediction .
docker run -p 8501:8501 drag-prediction
Business Vision
This digital twin reduces costly physical experiments by providing rapid, accurate drag predictions, enabling faster R&D cycles for industries such as aerospace, automotive, and maritime.

Future Work
Extend to multi-task prediction (e.g., lift and drag).
Integrate cloud deployment for scalability.
Enhance UI interactivity and real-time feedback.


---

# Conclusion

This comprehensive project code base:
- Processes raw image and CSV data with robust normalization (including for very low drag values).
- Implements two models (a baseline and an advanced CNN+LSTM pipeline).
- Splits data into training and test sets.
- Provides training, evaluation, interactive UI, and API components.
- Is containerized using Docker for professional deployment.

By following these guidelines and refining hyperparameters as needed, you should have a professional-level project that meets all the technical, business, and MLOps requirements—positioning you for a 100/100 in your machine learning project.

If you need any further refinements or additional clarifications, feel free to ask!
