import os
import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from code.utils.data_loader import ReynoldsDataLoader
from code.models.baseline_model import BaselineMLP
from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor

def compute_regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    # If y_true is constant, R² can be NaN; handle gracefully
    var = np.var(y_true)
    r2 = 1 - mse / var if var != 0 else float("nan")
    return mse, rmse, r2

def evaluate_baseline(baseline_model, test_dataset, device="cpu"):
    baseline_model.eval()
    x_list, y_list, y_pred_list = [], [], []
    with torch.no_grad():
        for x, y in test_dataset:
            x = x.unsqueeze(0).to(device)
            y_pred = baseline_model(x)
            x_list.append(x.cpu().numpy())
            y_list.append(y.cpu().numpy())
            y_pred_list.append(y_pred.cpu().numpy())

    y_true = np.concatenate(y_list).flatten()
    y_pred = np.concatenate(y_pred_list).flatten()
    mse, rmse, r2 = compute_regression_metrics(y_true, y_pred)
    return mse, rmse, r2

def evaluate_cnn_lstm(cnn_model, lstm_model, test_dataset, device="cpu"):
    """
    Evaluate CNN+LSTM on test set. We'll form a single sequence
    from the test set for simplicity. 
    """
    cnn_model.eval()
    lstm_model.eval()

    # Gather test data
    images_list = []
    drags_list = []
    for x, y in test_dataset:
        images_list.append(x.unsqueeze(0))
        drags_list.append(y)
    images_all = torch.cat(images_list, dim=0).to(device)  # shape: (N, 1, 64, 64)
    drags_all = torch.cat(drags_list, dim=0).to(device)    # shape: (N, 1)

    with torch.no_grad():
        # Extract latents
        latents = []
        for i in range(images_all.size(0)):
            latent, _ = cnn_model(images_all[i:i+1])
            latents.append(latent)
        latents_all = torch.cat(latents, dim=0)  # shape: (N, latent_dim)
        latents_all = latents_all.unsqueeze(1)   # (N, 1, latent_dim)
        # Predict with LSTM
        preds, _ = lstm_model(latents_all)       # (N, 1, 1)
        preds = preds.squeeze()                  # (N,)

    y_true = drags_all.cpu().numpy().flatten()
    y_pred = preds.cpu().numpy().flatten()
    mse, rmse, r2 = compute_regression_metrics(y_true, y_pred)
    return mse, rmse, r2, y_true, y_pred

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = config.get("device", "cpu")

    # Load data again
    loader = ReynoldsDataLoader(config)
    dataset = loader.load_dataset()

    # Combine
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

    # Split test portion. We'll do the same ratio as main or a fixed ratio
    from torch.utils.data import TensorDataset, random_split
    full_dataset = TensorDataset(images_combined, drags_combined)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Load models
    # Baseline
    sample_x, _ = test_dataset[0]
    input_dim = sample_x.numel()
    baseline_model = BaselineMLP(input_dim=input_dim).to(device)
    baseline_model.load_state_dict(torch.load(os.path.join("models", "baseline_mlp.pth"), map_location=device))

    # CNN
    from code.models.cnn_model import CNNDragPredictor
    cnn_model = CNNDragPredictor(latent_dim=config["cnn"].get("latent_dim", 64)).to(device)
    cnn_model.load_state_dict(torch.load(os.path.join("models", "cnn_drag_predictor.pth"), map_location=device))

    # LSTM
    from code.models.lstm_model import LSTMDragPredictor
    lstm_model = LSTMDragPredictor(
        input_size=config["lstm"].get("input_size", 64),
        hidden_size=config["lstm"].get("hidden_size", 32),
        num_layers=config["lstm"].get("num_layers", 1),
        output_size=config["lstm"].get("output_size", 1)
    ).to(device)
    lstm_model.load_state_dict(torch.load(os.path.join("models", "lstm_drag.pth"), map_location=device))

    # Evaluate Baseline
    mse_b, rmse_b, r2_b = evaluate_baseline(baseline_model, test_dataset, device=device)
    logging.info(f"[Baseline] MSE={mse_b:.6f}, RMSE={rmse_b:.6f}, R2={r2_b:.6f}")

    # Evaluate CNN+LSTM
    mse_c, rmse_c, r2_c, y_true, y_pred = evaluate_cnn_lstm(cnn_model, lstm_model, test_dataset, device=device)
    logging.info(f"[CNN+LSTM] MSE={mse_c:.6f}, RMSE={rmse_c:.6f}, R2={r2_c:.6f}")

    # Optional: plot comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(y_true, label="True Drag")
    plt.plot(y_pred, label="Predicted Drag")
    plt.title("CNN+LSTM Drag Prediction on Test Set")
    plt.xlabel("Time Step (Test Dataset Order)")
    plt.ylabel("Drag (scaled)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("evaluation_plots", exist_ok=True)
    plot_path = os.path.join("evaluation_plots", "cnn_lstm_evaluation.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved evaluation plot: {plot_path}")

if __name__ == "__main__":
    main()
