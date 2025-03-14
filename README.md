# Digital Twin for 2D Unsteady Flow: Drag Prediction

## 1. Introduction
This project develops a digital twin to predict the instantaneous drag force on a 2D flow around a sphere at various Reynolds numbers. We demonstrate:
- **Baseline Model**: A simple MLP that flattens images.
- **Advanced Model**: A CNN for spatial compression + LSTM for temporal evolution.

## 2. Repository Structure
project/ ├─ code/ │ ├─ utils/ │ ├─ models/ ├─ data/ ├─ main.py ├─ evaluate.py ├─ streamlit_app.py ├─ api.py ├─ Dockerfile ├─ requirements.txt └─ README.md

## 3. Installation & Usage
1. **Install Dependencies**:
```bash
pip install -r requirements.txt
python main.py
python evaluate.py
streamlit run streamlit_app.py
docker build -t drag-prediction .
docker run -p 8501:8501 drag-prediction
4. Technical Highlights
CNN extracts latent features from 2D snapshots.
LSTM evolves latent states for time-series drag prediction.
Configurable normalization (MinMax, Standard, Log) for extremely low drag values (10^-8 range).
5. Business Vision
This digital twin can replace or supplement costly CFD simulations or wind-tunnel tests, accelerating R&D in aerospace, automotive, or maritime industries.

6. Future Work
Add multi-task learning for lift + drag.
Incorporate domain adaptation across multiple Reynolds numbers.
Deploy on AWS/GCP with GPU acceleration.

---

# Conclusion & Next Steps

With this **comprehensive code base** and **project structure**, you now have:

1. A **Baseline** (simple MLP) and **Advanced** (CNN+LSTM) model to meet the “two models” requirement.  
2. A **Docker** setup for containerization and a **Flask API** or **Streamlit UI** for user interaction.  
3. A **clear MLOps pipeline** (data loading, training, evaluation, deployment).  
4. Documentation in **README.md** describing your approach and business vision.  
5. An optional path to **cloud deployment** on AWS/GCP for a truly production-ready solution.

Use this as a **foundation** to add further **presentation flair** (for the “wow factor”), refine hyperparameters, or integrate advanced domain knowledge (like special boundary conditions). This setup should **satisfy** the professor’s requirements for **technical complexity**, **software methodology**, **business aspect**, and **presentation**. Good luck with your digital twin project!
