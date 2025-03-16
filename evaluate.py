import os
import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, TensorDataset
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

def evaluate_baseline_model(baseline_model, test_dataset, device="cpu"):
    """
    Evaluate the Baseline MLP model on the test dataset.
    """
    baseline_model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for x, y in test_dataset:
            x = x.unsqueeze(0).to(device)
            y_pred = baseline_model(x)
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(y_pred.cpu().numpy())
    y_true = np.concatenate(y_true_list).flatten()
    y_pred = np.concatenate(y_pred_list).flatten()
    return compute_regression_metrics(y_true, y_pred)

def evaluate_cnn_lstm_model(cnn_model, lstm_model, test_dataset, device="cpu"):
    """
    Evaluate the CNN+LSTM pipeline on the test dataset.
    Returns:
        - metrics: Tuple (MSE, RMSE, R²)
        - y_true: True drag values (flattened)
        - y_pred: Predicted drag values (flattened)
    """
    cnn_model.eval()
    lstm_model.eval()
    images_list, drags_list = [], []
    for x, y in test_dataset:
        images_list.append(x.unsqueeze(0))
        drags_list.append(y)
    images_all = torch.cat(images_list, dim=0).to(device)
    drags_all = torch.cat(drags_list, dim=0).to(device).squeeze(-1)
    
    with torch.no_grad():
        latents = [cnn_model(images_all[i:i+1])[0] for i in range(images_all.size(0))]
    latents_all = torch.cat(latents, dim=0)
    latents_all = latents_all.unsqueeze(1)  # Reshape for LSTM: (seq_len, batch, input_size)
    with torch.no_grad():
        pred_seq, _ = lstm_model(latents_all)
        pred_seq = pred_seq.squeeze()
    
    y_true = drags_all.cpu().numpy().flatten()
    y_pred = pred_seq.cpu().numpy().flatten()
    metrics = compute_regression_metrics(y_true, y_pred)
    return metrics, y_true, y_pred

def robust_evaluation(dataset, baseline_model, cnn_model, lstm_model, device="cpu", folds=5):
    """
    Perform robust evaluation over multiple random splits.
    Returns average and standard deviation metrics for both models.
    """
    baseline_metrics = []
    cnn_lstm_metrics = []
    total_samples = len(dataset)
    
    for fold in range(folds):
        test_size = int(0.1 * total_samples)  # 10% test split per fold
        train_size = total_samples - test_size
        _, test_dataset = random_split(dataset, [train_size, test_size])
        
        mse_b, rmse_b, r2_b = evaluate_baseline_model(baseline_model, test_dataset, device=device)
        baseline_metrics.append((mse_b, rmse_b, r2_b))
        
        metrics_cnn, _, _ = evaluate_cnn_lstm_model(cnn_model, lstm_model, test_dataset, device=device)
        cnn_lstm_metrics.append(metrics_cnn)
        
        logging.info(f"Fold {fold+1}: Baseline MSE={mse_b:.6f}, CNN+LSTM MSE={metrics_cnn[0]:.6f}")
    
    baseline_metrics = np.array(baseline_metrics)
    cnn_lstm_metrics = np.array(cnn_lstm_metrics)
    
    results = {
        "baseline": {
            "MSE_mean": baseline_metrics[:, 0].mean(),
            "MSE_std": baseline_metrics[:, 0].std(),
            "RMSE_mean": baseline_metrics[:, 1].mean(),
            "RMSE_std": baseline_metrics[:, 1].std(),
            "R2_mean": baseline_metrics[:, 2].mean(),
            "R2_std": baseline_metrics[:, 2].std()
        },
        "cnn_lstm": {
            "MSE_mean": cnn_lstm_metrics[:, 0].mean(),
            "MSE_std": cnn_lstm_metrics[:, 0].std(),
            "RMSE_mean": cnn_lstm_metrics[:, 1].mean(),
            "RMSE_std": cnn_lstm_metrics[:, 1].std(),
            "R2_mean": cnn_lstm_metrics[:, 2].mean(),
            "R2_std": cnn_lstm_metrics[:, 2].std()
        }
    }
    return results

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    device = config.get("device", "cpu")
    
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
    
    # Create a final test set using a 10% split
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
    
    # Evaluate models
    mse_b, rmse_b, r2_b = evaluate_baseline_model(baseline_model, test_dataset, device=device)
    metrics_cnn, y_true, y_pred = evaluate_cnn_lstm_model(cnn_model, lstm_model, test_dataset, device=device)
    mse_c, rmse_c, r2_c = metrics_cnn
    
    logging.info(f"Baseline MLP - MSE: {mse_b:.6f}, RMSE: {rmse_b:.6f}, R²: {r2_b:.6f}")
    logging.info(f"CNN+LSTM   - MSE: {mse_c:.6f}, RMSE: {rmse_c:.6f}, R²: {r2_c:.6f}")
    
    results = robust_evaluation(full_dataset, baseline_model, cnn_model, lstm_model, device=device, folds=config["robust_evaluation"].get("folds", 5))
    logging.info("Robust Evaluation Results:")
    logging.info(f"Baseline MSE: {results['baseline']['MSE_mean']:.6f} ± {results['baseline']['MSE_std']:.6f}")
    logging.info(f"CNN+LSTM   MSE: {results['cnn_lstm']['MSE_mean']:.6f} ± {results['cnn_lstm']['MSE_std']:.6f}")
    
    # Plot predictions vs. true values for final test set
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True Drag", marker="o", linestyle="-")
    plt.plot(y_pred, label="Predicted Drag", marker="x", linestyle="--")
    plt.title("CNN+LSTM Drag Prediction on Final Test Set")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Drag (normalized)")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("evaluation_plots", exist_ok=True)
    plot_path = os.path.join("evaluation_plots", "cnn_lstm_evaluation.png")
    try:
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved evaluation plot: {plot_path}")
    except Exception as e:
        logging.error(f"Error saving evaluation plot: {e}")

if __name__ == "__main__":
    main()
