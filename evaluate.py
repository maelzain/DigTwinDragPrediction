import os
import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, random_split

from code.models.baseline_model import BaselineMLP
from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor
from code.utils.data_loader import ReynoldsDataLoader

def compute_regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    var = np.var(y_true)
    r2 = 1 - mse / var if var != 0 else float("nan")
    return mse, rmse, r2

def evaluate_baseline(baseline_model, test_dataset, device="cpu"):
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

def evaluate_cnn_lstm(cnn_model, lstm_model, test_dataset, device="cpu"):
    cnn_model.eval()
    lstm_model.eval()

    images_list, drags_list = [], []
    for x, y in test_dataset:
        images_list.append(x.unsqueeze(0))
        drags_list.append(y)
    images_all = torch.cat(images_list, dim=0).to(device)
    drags_all = torch.cat(drags_list, dim=0).to(device).squeeze(-1)

    with torch.no_grad():
        latents = []
        for i in range(images_all.size(0)):
            latent, _ = cnn_model(images_all[i:i+1])
            latents.append(latent)
        latents_all = torch.cat(latents, dim=0)  # (N, latent_dim)
        latents_all = latents_all.unsqueeze(1)    # (N, 1, latent_dim)
        preds, _ = lstm_model(latents_all)          # (N, 1, 1)
        preds = preds.squeeze()                     # (N,)
    y_true = drags_all.cpu().numpy().flatten()
    y_pred = preds.cpu().numpy().flatten()
    metrics = compute_regression_metrics(y_true, y_pred)
    return metrics, y_true, y_pred

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    device = config.get("device", "cpu")

    loader = ReynoldsDataLoader(config)
    dataset = loader.load_dataset()
    all_images = []
    all_drags = []
    for folder, data in dataset.items():
        images = data["images"]
        drag = data["drag"]
        if images is not None and images.numel() > 0 and drag is not None:
            all_images.append(images)
            all_drags.append(drag)
    images_combined = torch.cat(all_images, dim=0)
    drags_combined = torch.cat(all_drags, dim=0)

    full_dataset = TensorDataset(images_combined, drags_combined)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Load Baseline model
    sample_x, _ = test_dataset[0]
    input_dim = sample_x.numel()
    baseline_model = BaselineMLP(input_dim=input_dim).to(device)
    baseline_model.load_state_dict(torch.load(os.path.join("models", "baseline_mlp.pth"), map_location=device))
    mse_b, rmse_b, r2_b = evaluate_baseline(baseline_model, test_dataset, device=device)
    logging.info(f"[Baseline] MSE={mse_b:.6f}, RMSE={rmse_b:.6f}, R2={r2_b:.6f}")

    # Load CNN model
    cnn_model = CNNDragPredictor(latent_dim=config["cnn"].get("latent_dim", 64)).to(device)
    cnn_model.load_state_dict(torch.load(os.path.join("models", "cnn_drag_predictor.pth"), map_location=device))

    # Load LSTM model
    lstm_model = LSTMDragPredictor(
        input_size=config["lstm"].get("input_size", 64),
        hidden_size=config["lstm"].get("hidden_size", 32),
        num_layers=config["lstm"].get("num_layers", 1),
        output_size=config["lstm"].get("output_size", 1)
    ).to(device)
    lstm_model.load_state_dict(torch.load(os.path.join("models", "lstm_drag.pth"), map_location=device))

    (mse_c, rmse_c, r2_c), y_true, y_pred = evaluate_cnn_lstm(cnn_model, lstm_model, test_dataset, device=device)
    logging.info(f"[CNN+LSTM] MSE={mse_c:.6f}, RMSE={rmse_c:.6f}, R2={r2_c:.6f}")

    # Plot evaluation results
    plt.figure(figsize=(8, 5))
    plt.plot(y_true, label="True Drag")
    plt.plot(y_pred, label="Predicted Drag")
    plt.title("CNN+LSTM Drag Prediction on Test Set")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Drag (normalized)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("evaluation_plots", exist_ok=True)
    plot_path = os.path.join("evaluation_plots", "cnn_lstm_evaluation.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved evaluation plot: {plot_path}")

if __name__ == "__main__":
    main()
