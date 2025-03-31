import os
import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, TensorDataset
from scipy import stats

from code.utils.data_loader import ReynoldsDataLoader
from code.models.baseline_model import BaselineMLP
from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor

def compute_regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    var = np.var(y_true)
    r2 = 1 - mse / var if var != 0 else float("nan")
    return mse, rmse, r2

def compute_bias_variance(y_true, y_pred):
    bias = np.mean(y_pred - y_true)
    variance = np.var(y_pred)
    return bias, variance

def plot_true_vs_pred(y_true, y_pred, model_name="CNN+LSTM"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, label=model_name)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
    plt.xlabel("True Drag (normalized)")
    plt.ylabel("Predicted Drag (normalized)")
    plt.title(f"{model_name} Prediction Comparison")
    plt.legend()
    plt.tight_layout()
    os.makedirs("evaluation_plots", exist_ok=True)
    scatter_path = os.path.join("evaluation_plots", f"{model_name}_true_vs_pred.png")
    plt.savefig(scatter_path)
    plt.close()
    logging.info(f"Saved true vs. predicted scatter plot: {scatter_path}")

def plot_test_vs_true(y_true, y_pred, model_name="CNN+LSTM"):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True Drag", marker='o', linestyle='-')
    plt.plot(y_pred, label="Predicted Drag", marker='x', linestyle='--', alpha=0.8)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Drag (normalized)")
    plt.title(f"{model_name} - Test Samples: True vs. Predicted Drag")
    plt.legend()
    plt.tight_layout()
    os.makedirs("evaluation_plots", exist_ok=True)
    line_path = os.path.join("evaluation_plots", f"{model_name}_test_vs_true.png")
    plt.savefig(line_path)
    plt.close()
    logging.info(f"Saved test vs. true line plot: {line_path}")

def plot_residuals(y_true, y_pred, model_name="CNN+LSTM"):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.title(f"{model_name} Residual Histogram")
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    os.makedirs("evaluation_plots", exist_ok=True)
    hist_path = os.path.join("evaluation_plots", f"{model_name}_residual_hist.png")
    plt.savefig(hist_path)
    plt.close()
    logging.info(f"Saved residual histogram: {hist_path}")
    
    plt.figure(figsize=(8, 5))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"{model_name} Q-Q Plot of Residuals")
    plt.tight_layout()
    qq_path = os.path.join("evaluation_plots", f"{model_name}_qq_plot.png")
    plt.savefig(qq_path)
    plt.close()
    logging.info(f"Saved Q-Q plot: {qq_path}")

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
    bv_path = os.path.join("evaluation_plots", "bias_variance_comparison.png")
    plt.savefig(bv_path)
    plt.close()
    logging.info(f"Saved bias and variance comparison plot: {bv_path}")

def paired_t_test(y_true, y_pred_baseline, y_pred_cnn):
    errors_baseline = np.abs(y_true - y_pred_baseline)
    errors_cnn = np.abs(y_true - y_pred_cnn)
    t_stat, p_val = stats.ttest_rel(errors_baseline, errors_cnn)
    return t_stat, p_val

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Load configuration from config.yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device(config.get("device", "cpu"))
    
    # Load dataset using ReynoldsDataLoader
    from code.utils.data_loader import ReynoldsDataLoader
    loader = ReynoldsDataLoader(config)
    data_dict = loader.load_dataset()
    all_images, all_drags = [], []
    for folder, data in data_dict.items():
        if data["images"] is not None and data["images"].numel() > 0 and data["drag"] is not None:
            all_images.append(data["images"])
            all_drags.append(data["drag"])
    images_combined = torch.cat(all_images, dim=0)
    drags_combined = torch.cat(all_drags, dim=0)
    full_dataset = TensorDataset(images_combined, drags_combined)
    
    total_samples = len(full_dataset)
    test_size = int(config["split"]["test_ratio"] * total_samples)
    train_size = total_samples - test_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])
    
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
        logging.error(f"Error loading models: {e}")
        return
    
    # Evaluate Baseline Model on test set
    baseline_preds = []
    baseline_trues = []
    baseline_model.eval()
    with torch.no_grad():
        for x, y in test_dataset:
            x = x.unsqueeze(0).to(device)
            pred = baseline_model(x)
            baseline_preds.append(pred.cpu().numpy())
            baseline_trues.append(y.cpu().numpy())
    baseline_trues = np.concatenate(baseline_trues).flatten()
    baseline_preds = np.concatenate(baseline_preds).flatten()
    mse_b, rmse_b, r2_b = compute_regression_metrics(baseline_trues, baseline_preds)
    bias_b, var_b = compute_bias_variance(baseline_trues, baseline_preds)
    logging.info(f"Final Test Set - Baseline MLP: MSE: {mse_b:.6f}, RMSE: {rmse_b:.6f}, R²: {r2_b:.6f}, Bias: {bias_b:.6f}, Variance: {var_b:.6f}")
    
    # Evaluate CNN+LSTM on test set
    images_list, drags_list = [], []
    for x, y in test_dataset:
        images_list.append(x.unsqueeze(0))
        drags_list.append(y)
    images_all = torch.cat(images_list, dim=0).to(device)
    drags_all = torch.cat(drags_list, dim=0).to(device).squeeze(-1)
    cnn_model.eval()
    lstm_model.eval()
    with torch.no_grad():
        latents = [cnn_model(images_all[i:i+1])[0] for i in range(images_all.size(0))]
    latents_all = torch.cat(latents, dim=0).unsqueeze(1)
    with torch.no_grad():
        pred_seq, _ = lstm_model(latents_all)
        pred_seq = pred_seq.squeeze()
        if pred_seq.dim() == 0:
            pred_seq = pred_seq.unsqueeze(0)
    cnn_trues = drags_all.cpu().numpy().flatten()
    cnn_preds = pred_seq.cpu().numpy().flatten()
    mse_c, rmse_c, r2_c = compute_regression_metrics(cnn_trues, cnn_preds)
    bias_c, var_c = compute_bias_variance(cnn_trues, cnn_preds)
    logging.info(f"Final Test Set - CNN+LSTM: MSE: {mse_c:.6f}, RMSE: {rmse_c:.6f}, R²: {r2_c:.6f}, Bias: {bias_c:.6f}, Variance: {var_c:.6f}")
    
    # Plot visualizations
    plot_true_vs_pred(cnn_trues, cnn_preds, model_name="CNN+LSTM")
    plot_test_vs_true(cnn_trues, cnn_preds, model_name="CNN+LSTM")
    plot_residuals(cnn_trues, cnn_preds, model_name="CNN+LSTM")
    
    model_names = ["Baseline", "CNN+LSTM"]
    bias_values = [bias_b, bias_c]
    variance_values = [var_b, var_c]
    plot_bias_variance(bias_values, variance_values, model_names)
    
    t_stat, p_val = paired_t_test(baseline_trues, baseline_preds, cnn_preds)
    logging.info(f"Paired t-test on absolute errors: t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}")

if __name__ == "__main__":
    main()
