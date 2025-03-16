import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_csv_file(csv_path):
    """
    Load a CSV file into a DataFrame.
    Logs and returns None if file not found or an error occurs.
    """
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded CSV file: {csv_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file {csv_path}: {e}")
        return None

def normalize_drag_forces(df, column_name="cd", method="minmax"):
    """
    Normalize drag force values in the specified column.
    For very low, strictly positive drag forces, min–max scaling is preferred.
    Supported methods:
      - 'minmax': scales data to [0, 1]
      - 'standard': applies Z-score normalization (may yield negative values)
      - 'log': applies logarithmic transformation
    """
    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' not found in dataframe.")
        return df

    if method == "log":
        df[column_name] = np.log(df[column_name] + 1e-10)
        logging.info(f"Applied log transform to '{column_name}'.")
    elif method == "minmax":
        scaler = MinMaxScaler()
        df[column_name] = scaler.fit_transform(df[[column_name]])
        logging.info(f"Normalized column '{column_name}' using minmax scaler.")
    else:
        scaler = StandardScaler()
        df[column_name] = scaler.fit_transform(df[[column_name]])
        logging.info(f"Normalized column '{column_name}' using standard scaler.")
    return df

def preprocess_drag_csv(csv_path, column_name="cd", norm_method="minmax"):
    """
    Load and preprocess the CSV file containing drag forces.
    By default, applies min–max normalization.
    """
    df = load_csv_file(csv_path)
    if df is not None:
        df = normalize_drag_forces(df, column_name, method=norm_method)
    return df
