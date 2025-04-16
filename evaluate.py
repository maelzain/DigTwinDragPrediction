import os
import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import random_split, TensorDataset, DataLoader
from scipy import stats
from itertools import product

# Custom modules
from code.utils.data_loader import ReynoldsDataLoader
from code.models.baseline_model import BaselineMLP
from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor


# ===============================
# Additional Metric Computations
# ===============================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute standard regression metrics: MSE, RMSE, R²,
    plus additional metrics like MAE and Pearson correlation.
    Also compute the Mean Absolute Percentage Error (MAPE) in percent.
    Returns a dictionary with all metrics.
    """
    errors = y_true - y_pred
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    var = np.var(y_true)
    r2 = 1 - mse / var if var != 0 else float("nan")
    try:
        pearson_corr = stats.pearsonr(y_true, y_pred)[0]
    except Exception:
        pearson_corr = float("nan")
    mask = (y_true != 0)
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    else:
        mape = float("nan")
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "PearsonCorr": pearson_corr,
        "MAPE(%)": mape
    }

def compute_bias_variance(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute bias (mean error) and variance of the predictions.
    """
    bias = np.mean(y_pred - y_true)
    variance = np.var(y_pred)
    return bias, variance


# ===============================
# Plotting / Visualization
# ===============================

def plot_parity(y_true, y_pred, model_name="CNN + LSTM"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, label=model_name)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
    plt.xlabel("True Drag (physical units)")
    plt.ylabel("Predicted Drag (physical units)")
    plt.title(f"{model_name} Parity Plot")
    plt.legend()
    plt.tight_layout()
    os.makedirs("evaluation_plots", exist_ok=True)
    fname = os.path.join("evaluation_plots", f"{model_name}_parity_plot.png")
    plt.savefig(fname)
    plt.close()
    logging.info(f"Saved parity plot: {fname}")

def plot_test_samples(y_true, y_pred, model_name="CNN + LSTM"):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, 'o-', label="True Drag", alpha=0.7)
    plt.plot(y_pred, 'x--', label="Predicted Drag", alpha=0.7)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Drag (physical units)")
    plt.title(f"{model_name} - Test Sample Comparison")
    plt.legend()
    plt.tight_layout()
    os.makedirs("evaluation_plots", exist_ok=True)
    fname = os.path.join("evaluation_plots", f"{model_name}_test_samples.png")
    plt.savefig(fname)
    plt.close()
    logging.info(f"Saved test sample comparison plot: {fname}")

def plot_advanced_residual_analysis(y_true, y_pred, model_name="CNN + LSTM"):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{model_name} - Advanced Residual Analysis", fontsize=14)
    axes[0, 0].scatter(y_pred, residuals, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel("Predicted Drag (physical units)")
    axes[0, 0].set_ylabel("Residual (True - Predicted)")
    axes[0, 0].set_title("Residual vs. Predicted")
    sample_index = np.arange(len(y_true))
    axes[0, 1].scatter(sample_index, residuals, alpha=0.7)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel("Test Sample Index")
    axes[0, 1].set_ylabel("Residual (True - Predicted)")
    axes[0, 1].set_title("Residual vs. Sample Index")
    axes[1, 0].hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    axes[1, 0].set_xlabel("Residual (True - Predicted)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Residual Histogram")
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot of Residuals")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs("evaluation_plots", exist_ok=True)
    fname = os.path.join("evaluation_plots", f"{model_name}_advanced_residual_analysis.png")
    plt.savefig(fname)
    plt.close()
    logging.info(f"Saved advanced residual analysis figure: {fname}")

def plot_bias_variance(bias_values, variance_values, model_names):
    x = np.arange(len(model_names))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, bias_values, width, label='Bias')
    plt.bar(x + width/2, variance_values, width, label='Variance')
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title('Bias and Variance Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    plt.tight_layout()
    os.makedirs("evaluation_plots", exist_ok=True)
    fname = os.path.join("evaluation_plots", "bias_variance_comparison.png")
    plt.savefig(fname)
    plt.close()
    logging.info(f"Saved bias and variance comparison: {fname}")

def plot_time_series_generated(y_true, y_pred, model_name="CNN + LSTM", start=10, end=3000):
    timesteps = np.linspace(start, end, num=len(y_true))
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, y_true, 'o-', label="True Drag", color='blue')
    plt.plot(timesteps, y_pred, 'x--', label="Predicted Drag", color='red')
    plt.xlabel("Time Step (Generated)")
    plt.ylabel("Drag (physical units)")
    plt.title(f"{model_name} - Drag vs. Generated Time")
    plt.legend()
    plt.tight_layout()
    os.makedirs("evaluation_plots", exist_ok=True)
    fname = os.path.join("evaluation_plots", f"{model_name}_drag_vs_time_generated.png")
    plt.savefig(fname)
    plt.close()
    logging.info(f"Saved time series plot (generated): {fname}")


# ===============================
# Helper: Convert Normalized to Physical Drag
# ===============================
def convert_to_physical(normalized_array, group_list, drag_ranges):
    """
    Given a 1D numpy array of normalized drag values and a corresponding
    list of group labels (e.g., "re37", "re75", etc.), convert to physical units.
    """
    physical_values = np.zeros_like(normalized_array)
    for i, group in enumerate(group_list):
        if group in drag_ranges:
            rmin, rmax = drag_ranges[group]
        else:
            rmin, rmax = 0, 1  # Fallback
        physical_values[i] = normalized_array[i] * (rmax - rmin) + rmin
    return physical_values


# ===============================
# Evaluate Pipeline
# ===============================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load configuration and set device
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device(config.get("device", "cpu"))
    
    # -------------------------------
    # Data Loading with Group Labels
    # -------------------------------
    loader = ReynoldsDataLoader(config)
    data_dict = loader.load_dataset()
    
    all_images = []
    all_drags = []
    all_groups = []  # To store the Reynolds group label for each sample
    
    for folder, data in data_dict.items():
        if data["images"] is not None and data["images"].numel() > 0 and data["drag"] is not None:
            all_images.append(data["images"])
            all_drags.append(data["drag"])
            num_samples = data["images"].size(0)
            all_groups.extend([folder] * num_samples)
    
    images_combined = torch.cat(all_images, dim=0)
    drags_combined = torch.cat(all_drags, dim=0)
    
    # Create full dataset (group labels are maintained externally)
    full_dataset = TensorDataset(images_combined, drags_combined)
    
    # -------------------------------
    # Split into Train and Test Sets
    # -------------------------------
    total_samples = len(full_dataset)
    test_size = int(config["split"]["test_ratio"] * total_samples)
    train_size = total_samples - test_size
    train_subset, test_subset = random_split(full_dataset, [train_size, test_size])
    
    # Recover test group labels using indices (assuming random_split provides .indices)
    try:
        test_indices = test_subset.indices
    except AttributeError:
        test_indices = list(range(train_size, total_samples))
    test_groups = [all_groups[i] for i in test_indices]
    
    # -------------------------------
    # Initialize and Load Models
    # -------------------------------
    sample_x, _ = full_dataset[0]
    input_dim = sample_x.numel()
    baseline_model = BaselineMLP(input_dim=input_dim, hidden_dim=config["baseline"]["hidden_dim"]).to(device)
    cnn_model = CNNDragPredictor(latent_dim=config["cnn"]["latent_dim"]).to(device)
    lstm_model = LSTMDragPredictor(
        input_size=config["lstm"]["input_size"],
        hidden_size=config["lstm"]["hidden_size"],
        num_layers=config["lstm"]["num_layers"],
        output_size=config["lstm"]["output_size"]
    ).to(device)
    
    try:
        baseline_model.load_state_dict(torch.load(os.path.join(config["model_dir"], "baseline_mlp.pth"), map_location=device))
        cnn_model.load_state_dict(torch.load(os.path.join(config["model_dir"], "cnn_drag_predictor.pth"), map_location=device))
        lstm_model.load_state_dict(torch.load(os.path.join(config["model_dir"], "lstm_drag.pth"), map_location=device))
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        return
    
    # -------------------------------
    # Evaluate Baseline Model
    # -------------------------------
    baseline_trues, baseline_preds = [], []
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    baseline_model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = baseline_model(x)
            baseline_preds.append(pred.cpu().numpy())
            baseline_trues.append(y.cpu().numpy())
    baseline_trues = np.concatenate(baseline_trues).flatten()
    baseline_preds = np.concatenate(baseline_preds).flatten()
    
    # -------------------------------
    # Evaluate CNN + LSTM
    # -------------------------------
    images_list, drags_list = [], []
    for x, y in test_subset:
        images_list.append(x.unsqueeze(0))
        drags_list.append(y)
    images_all = torch.cat(images_list, dim=0).to(device)
    drags_all = torch.cat(drags_list, dim=0).to(device).squeeze(-1)
    
    cnn_model.eval()
    lstm_model.eval()
    with torch.no_grad():
        latents = []
        for i in range(images_all.size(0)):
            latent, _ = cnn_model(images_all[i:i+1])
            latents.append(latent)
        latents_all = torch.cat(latents, dim=0).unsqueeze(1)
        pred_seq, _ = lstm_model(latents_all)
        pred_seq = pred_seq.squeeze()
        if pred_seq.dim() == 0:
            pred_seq = pred_seq.unsqueeze(0)
    cnn_trues = drags_all.cpu().numpy().flatten()
    cnn_preds = pred_seq.cpu().numpy().flatten()
    
    # -------------------------------
    # Conversion: Normalized -> Physical Drag
    # -------------------------------
    drag_ranges = config.get("drag_ranges", {})
    baseline_trues_physical = convert_to_physical(baseline_trues, test_groups, drag_ranges)
    baseline_preds_physical = convert_to_physical(baseline_preds, test_groups, drag_ranges)
    cnn_trues_physical = convert_to_physical(cnn_trues, test_groups, drag_ranges)
    cnn_preds_physical = convert_to_physical(cnn_preds, test_groups, drag_ranges)
    
    # -------------------------------
    # Compute Metrics Using Physical Drag Values
    # -------------------------------
    baseline_metrics = compute_regression_metrics(baseline_trues_physical, baseline_preds_physical)
    baseline_bias, baseline_var = compute_bias_variance(baseline_trues_physical, baseline_preds_physical)
    cnn_metrics = compute_regression_metrics(cnn_trues_physical, cnn_preds_physical)
    cnn_bias, cnn_var = compute_bias_variance(cnn_trues_physical, cnn_preds_physical)
    
    logging.info("=== Baseline MLP Performance (Physical Drag Units) ===")
    for k, v in baseline_metrics.items():
        logging.info(f"{k}: {v:.6f}")
    logging.info(f"Bias: {baseline_bias:.6f}, Variance: {baseline_var:.6f}")
    logging.info(f"Baseline MAPE: {baseline_metrics.get('MAPE(%)', float('nan')):.6f} %")
    
    logging.info("=== CNN + LSTM Performance (Physical Drag Units) ===")
    for k, v in cnn_metrics.items():
        logging.info(f"{k}: {v:.6f}")
    logging.info(f"Bias: {cnn_bias:.15f}, Variance: {cnn_var:.15f}")
    logging.info(f"CNN + LSTM MAPE: {cnn_metrics.get('MAPE(%)', float('nan')):.6f} %")
    
    # -------------------------------
    # Generate Plots Using Physical Drag Units
    # -------------------------------
    plot_parity(cnn_trues_physical, cnn_preds_physical, model_name="CNN + LSTM")
    plot_test_samples(cnn_trues_physical, cnn_preds_physical, model_name="CNN + LSTM")
    plot_advanced_residual_analysis(cnn_trues_physical, cnn_preds_physical, model_name="CNN + LSTM")
    plot_bias_variance([baseline_bias, cnn_bias], [baseline_var, cnn_var], ["Baseline", "CNN + LSTM"])
    plot_time_series_generated(cnn_trues_physical, cnn_preds_physical, model_name="CNN + LSTM", start=10, end=3000)
    
    # -------------------------------
    # Statistical Test (Paired t-test)
    # -------------------------------
    t_stat, p_val = stats.ttest_rel(np.abs(baseline_trues_physical - baseline_preds_physical),
                                    np.abs(cnn_trues_physical - cnn_preds_physical))
    logging.info(f"Paired t-test (Baseline vs. CNN+LSTM) => t-stat: {t_stat:.4f}, p-value: {p_val:.4e}")


if __name__ == "__main__":
    main()
