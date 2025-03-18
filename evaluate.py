import os
import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, TensorDataset
from scipy import stats  # for statistical hypothesis testing and Q-Q plot
from code.utils.data_loader import ReynoldsDataLoader
from code.models.baseline_model import BaselineMLP
from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor

def compute_regression_metrics(y_true, y_pred):
    """
    Compute regression metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    and Coefficient of Determination (R²).
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    var = np.var(y_true)
    r2 = 1 - mse / var if var != 0 else float("nan")
    return mse, rmse, r2

def robust_evaluation(dataset, baseline_model, cnn_model, lstm_model, device="cpu", folds=5):
    """
    Perform robust evaluation over multiple random splits (K-fold).
    Splits each fold with 90% training and 10% testing.
    Computes error metrics, bias (mean error), and variance for each model.
    This approach follows the best practice of using K-fold testing to assess average model performance.
    """
    baseline_metrics = []
    cnn_lstm_metrics = []
    baseline_biases = []
    baseline_variances = []
    cnn_biases = []
    cnn_variances = []
    total_samples = len(dataset)
    
    for fold in range(folds):
        test_size = int(0.1 * total_samples)  # 10% test split per fold
        train_size = total_samples - test_size
        _, test_dataset = random_split(dataset, [train_size, test_size])
        
        # --- Baseline Model Evaluation ---
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
        baseline_metrics.append((mse_b, rmse_b, r2_b))
        bias_b = np.mean(baseline_preds - baseline_trues)
        var_b = np.var(baseline_preds)
        baseline_biases.append(bias_b)
        baseline_variances.append(var_b)
        
        # --- CNN+LSTM Evaluation ---
        images_list, drags_list = [], []
        for x, y in test_dataset:
            images_list.append(x.unsqueeze(0))
            drags_list.append(y)
        images_all = torch.cat(images_list, dim=0).to(device)  # shape: (N, 1, 64, 64)
        drags_all = torch.cat(drags_list, dim=0).to(device).squeeze(-1)
        
        cnn_model.eval()
        lstm_model.eval()
        with torch.no_grad():
            latents = [cnn_model(images_all[i:i+1])[0] for i in range(images_all.size(0))]
        latents_all = torch.cat(latents, dim=0)  # shape: (N, latent_dim)
        latents_all = latents_all.unsqueeze(1)   # reshape for LSTM: (seq_len, batch, input_size)
        with torch.no_grad():
            pred_seq, _ = lstm_model(latents_all)
            pred_seq = pred_seq.squeeze()
        cnn_trues = drags_all.cpu().numpy().flatten()
        cnn_preds = pred_seq.cpu().numpy().flatten()
        mse_c, rmse_c, r2_c = compute_regression_metrics(cnn_trues, cnn_preds)
        cnn_lstm_metrics.append((mse_c, rmse_c, r2_c))
        bias_c = np.mean(cnn_preds - cnn_trues)
        var_c = np.var(cnn_preds)
        cnn_biases.append(bias_c)
        cnn_variances.append(var_c)
        
        logging.info(f"Fold {fold+1}: Baseline MSE={mse_b:.6f}, CNN+LSTM MSE={mse_c:.6f}")
    
    baseline_metrics = np.array(baseline_metrics)
    cnn_lstm_metrics = np.array(cnn_lstm_metrics)
    
    results = {
        "baseline": {
            "MSE_mean": baseline_metrics[:, 0].mean(),
            "MSE_std": baseline_metrics[:, 0].std(),
            "RMSE_mean": baseline_metrics[:, 1].mean(),
            "RMSE_std": baseline_metrics[:, 1].std(),
            "R2_mean": baseline_metrics[:, 2].mean(),
            "R2_std": baseline_metrics[:, 2].std(),
            "bias_mean": np.mean(baseline_biases),
            "variance_mean": np.mean(baseline_variances)
        },
        "cnn_lstm": {
            "MSE_mean": cnn_lstm_metrics[:, 0].mean(),
            "MSE_std": cnn_lstm_metrics[:, 0].std(),
            "RMSE_mean": cnn_lstm_metrics[:, 1].mean(),
            "RMSE_std": cnn_lstm_metrics[:, 1].std(),
            "R2_mean": cnn_lstm_metrics[:, 2].mean(),
            "R2_std": cnn_lstm_metrics[:, 2].std(),
            "bias_mean": np.mean(cnn_biases),
            "variance_mean": np.mean(cnn_variances)
        }
    }
    return results

def plot_bias_variance(results):
    """
    Generate a bar chart comparing the bias and variance of the baseline and CNN+LSTM models.
    """
    labels = ['Baseline', 'CNN+LSTM']
    bias = [results['baseline']['bias_mean'], results['cnn_lstm']['bias_mean']]
    variance = [results['baseline']['variance_mean'], results['cnn_lstm']['variance_mean']]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, bias, width, label='Bias (Mean Error)')
    ax.bar(x + width/2, variance, width, label='Variance of Predictions')
    
    ax.set_ylabel('Value')
    ax.set_title('Bias and Variance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    os.makedirs("evaluation_plots", exist_ok=True)
    plot_path = os.path.join("evaluation_plots", "bias_variance_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved bias and variance comparison plot: {plot_path}")

def plot_residuals(y_true, y_pred, model_name="CNN+LSTM"):
    """
    Plot residual histogram and Q-Q plot for the model predictions.
    """
    residuals = y_true - y_pred
    
    # Residual Histogram
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
    
    # Q-Q Plot
    plt.figure(figsize=(8, 5))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"{model_name} Q-Q Plot of Residuals")
    plt.tight_layout()
    qq_path = os.path.join("evaluation_plots", f"{model_name}_qq_plot.png")
    plt.savefig(qq_path)
    plt.close()
    logging.info(f"Saved Q-Q plot: {qq_path}")

def plot_true_vs_pred(y_true, y_pred, model_name="CNN+LSTM"):
    """
    Plot a scatter plot of true vs. predicted values with a 45° line.
    """
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

def paired_t_test(y_true, y_pred_baseline, y_pred_cnn):
    """
    Perform a paired t-test on the absolute errors of two models.
    Returns the t-statistic and p-value.
    """
    errors_baseline = np.abs(y_true - y_pred_baseline)
    errors_cnn = np.abs(y_true - y_pred_cnn)
    t_stat, p_val = stats.ttest_rel(errors_baseline, errors_cnn)
    return t_stat, p_val

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    device = config.get("device", "cpu")
    
    # Load dataset using ReynoldsDataLoader
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
    
    # Split the data: use 10% as a final test set for plotting predictions
    total_samples = len(full_dataset)
    test_size = int(config["split"].get("test_ratio", 0.1) * total_samples)
    train_size = total_samples - test_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Load models
    sample_x, _ = full_dataset[0]
    input_dim = sample_x.numel()
    baseline_model = BaselineMLP(input_dim=input_dim).to(device)
    cnn_model = CNNDragPredictor(latent_dim=config["cnn"].get("latent_dim", 128)).to(device)
    lstm_model = LSTMDragPredictor(
        input_size=config["lstm"].get("input_size", 128),
        hidden_size=config["lstm"].get("hidden_size", 64),
        num_layers=config["lstm"].get("num_layers", 2),
        output_size=config["lstm"].get("output_size", 1)
    ).to(device)
    
    try:
        baseline_model.load_state_dict(torch.load(os.path.join("models", "baseline_mlp.pth"), map_location=device))
        cnn_model.load_state_dict(torch.load(os.path.join("models", "cnn_drag_predictor.pth"), map_location=device))
        lstm_model.load_state_dict(torch.load(os.path.join("models", "lstm_drag.pth"), map_location=device))
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return
    
    # Evaluate Baseline on final test set for visualization
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
    logging.info(f"Final Test Set - Baseline MLP: MSE: {mse_b:.6f}, RMSE: {rmse_b:.6f}, R²: {r2_b:.6f}")
    
    # Evaluate CNN+LSTM on final test set for visualization
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
    latents_all = torch.cat(latents, dim=0)
    latents_all = latents_all.unsqueeze(1)
    with torch.no_grad():
        pred_seq, _ = lstm_model(latents_all)
        pred_seq = pred_seq.squeeze()
    cnn_trues = drags_all.cpu().numpy().flatten()
    cnn_preds = pred_seq.cpu().numpy().flatten()
    mse_c, rmse_c, r2_c = compute_regression_metrics(cnn_trues, cnn_preds)
    logging.info(f"Final Test Set - CNN+LSTM: MSE: {mse_c:.6f}, RMSE: {rmse_c:.6f}, R²: {r2_c:.6f}")
    
    # Plot true vs predicted scatter plot for CNN+LSTM
    plot_true_vs_pred(cnn_trues, cnn_preds, model_name="CNN+LSTM")
    
    # Plot residuals for CNN+LSTM
    plot_residuals(cnn_trues, cnn_preds, model_name="CNN+LSTM")
    
    # Robust evaluation using k-fold cross-validation
    robust_results = robust_evaluation(full_dataset, baseline_model, cnn_model, lstm_model,
                                       device=device, folds=config["robust_evaluation"].get("folds", 5))
    logging.info("Robust Evaluation Results:")
    logging.info(f"Baseline MSE: {robust_results['baseline']['MSE_mean']:.6f} ± {robust_results['baseline']['MSE_std']:.6f}")
    logging.info(f"CNN+LSTM   MSE: {robust_results['cnn_lstm']['MSE_mean']:.6f} ± {robust_results['cnn_lstm']['MSE_std']:.6f}")
    logging.info(f"Baseline Bias: {robust_results['baseline']['bias_mean']:.6e}, Variance: {robust_results['baseline']['variance_mean']:.6e}")
    logging.info(f"CNN+LSTM Bias: {robust_results['cnn_lstm']['bias_mean']:.6e}, Variance: {robust_results['cnn_lstm']['variance_mean']:.6e}")
    
    # Plot bias and variance comparison
    plot_bias_variance(robust_results)
    
    # Statistical Hypothesis Testing: Paired t-test on absolute errors between Baseline and CNN+LSTM predictions
    t_stat, p_val = paired_t_test(baseline_trues, baseline_preds, cnn_preds)
    logging.info(f"Paired t-test on absolute errors: t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}")
    
    # Final plot: Predictions vs. True Values for CNN+LSTM on Final Test Set (line plot)
    plt.figure(figsize=(10, 6))
    plt.plot(cnn_trues, label="True Drag", marker="o", linestyle="-")
    plt.plot(cnn_preds, label="Predicted Drag", marker="x", linestyle="--")
    plt.title("CNN+LSTM Drag Prediction on Final Test Set")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Drag (normalized)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("evaluation_plots", exist_ok=True)
    final_plot_path = os.path.join("evaluation_plots", "cnn_lstm_final_comparison.png")
    try:
        plt.savefig(final_plot_path)
        plt.close()
        logging.info(f"Saved final evaluation plot: {final_plot_path}")
    except Exception as e:
        logging.error(f"Error saving final evaluation plot: {e}")

if __name__ == "__main__":
    main()
