#!/bin/bash
# Start the Flask API in the background
python api.py &

# Start the Streamlit app (this process will be the foreground one)
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
