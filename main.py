import os
import logging
import yaml
import torch
from torch.utils.data import TensorDataset, random_split, Subset
from itertools import product

from code.utils.data_loader import ReynoldsDataLoader
from train import train_baseline_model, train_cnn_model, train_lstm_model

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"Config loaded with keys: {list(config.keys())}")
    return config

def split_full_dataset(images, drags, train_ratio, test_ratio, eval_ratio):
    total_samples = images.shape[0]
    test_size = int(test_ratio * total_samples)
    train_size = total_samples - test_size
    full_dataset = TensorDataset(images, drags)
    train_dataset, final_test_dataset = random_split(full_dataset, [train_size, test_size])
    logging.info(f"Training dataset: {len(train_dataset)} samples, Final test dataset: {len(final_test_dataset)} samples.")
    return train_dataset, final_test_dataset

def grid_search_cnn_lstm(train_dataset, config, device, folds=5):
    param_grid = {
        "cnn.learning_rate": [0.0001, 0.0005],
        "cnn.epochs": [30, 50],
        "cnn.batch_size": [10, 20],
        "cnn.latent_dim": [128, 256],
        "lstm.learning_rate": [0.0001, 0.0005],
        "lstm.epochs": [50, 100],
        "lstm.batch_size": [32, 64],
        "lstm.hidden_size": [64, 128]
    }
    best_score = float('inf')
    best_params = {}
    
    total_samples = len(train_dataset)
    indices = list(range(total_samples))
    fold_size = total_samples // folds
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    
    for combo in product(*values):
        for key, value in zip(keys, combo):
            section, param = key.split('.')
            config[section][param] = value
        # Ensure LSTM input_size matches CNN latent_dim
        config["lstm"]["input_size"] = config["cnn"]["latent_dim"]

        cv_losses = []
        for fold in range(folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold != folds - 1 else total_samples
            val_indices = indices[val_start:val_end]
            train_indices = indices[:val_start] + indices[val_end:]
            
            cv_train_subset = Subset(train_dataset, train_indices)
            cv_val_subset = Subset(train_dataset, val_indices)
            
            cnn_model = train_cnn_model(cv_train_subset, config, device=device)
            lstm_model = train_lstm_model(cnn_model, cv_train_subset, config, device=device)
            
            cnn_model.eval()
            lstm_model.eval()
            criterion = torch.nn.MSELoss()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in cv_val_subset:
                    x = x.unsqueeze(0).to(device)
                    y = y.to(device)
                    latent, _ = cnn_model(x)
                    latent_seq = latent.unsqueeze(1)
                    pred_seq, _ = lstm_model(latent_seq)
                    pred = pred_seq.squeeze()
                    if pred.dim() == 0:
                        pred = pred.unsqueeze(0)
                    if y.dim() == 0:
                        y = y.unsqueeze(0)
                    loss = criterion(pred, y)
                    val_loss += loss.item()
            avg_loss = val_loss / len(cv_val_subset)
            cv_losses.append(avg_loss)
        
        mean_cv_loss = sum(cv_losses) / len(cv_losses)
        logging.info(f"Grid Search CNN+LSTM: params={combo} => Mean CV Loss: {mean_cv_loss:.6f}")
        if mean_cv_loss < best_score:
            best_score = mean_cv_loss
            best_params = {k: v for k, v in zip(keys, combo)}
    
    return best_params, best_score

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(42)
    
    config = load_config()
    device = torch.device(config.get("device", "cpu"))
    
    from code.utils.data_loader import ReynoldsDataLoader
    loader = ReynoldsDataLoader(config)
    data_dict = loader.load_dataset()
    all_images, all_drags = [], []
    for folder, data in data_dict.items():
        if data["images"] is not None and data["images"].numel() > 0 and data["drag"] is not None:
            all_images.append(data["images"])
            all_drags.append(data["drag"])
    if not all_images:
        logging.error("No valid images found in dataset.")
        return
    
    images_combined = torch.cat(all_images, dim=0)
    drags_combined = torch.cat(all_drags, dim=0)
    
    train_dataset, final_test_dataset = split_full_dataset(
        images_combined, drags_combined,
        config["split"]["train_ratio"],
        config["split"]["test_ratio"],
        config["split"]["eval_ratio"]
    )
    
    if config.get("grid_search", False):
        logging.info("Starting grid search for CNN+LSTM hyperparameters.")
        best_params, best_score = grid_search_cnn_lstm(train_dataset, config, device=device,
                                                       folds=config["robust_evaluation"]["folds"])
        logging.info(f"Best CNN+LSTM hyperparameters: {best_params} with CV Loss: {best_score:.6f}")
        for key, value in best_params.items():
            section, param = key.split('.')
            config[section][param] = value
    
    from train import train_baseline_model, train_cnn_model, train_lstm_model
    cnn_model = train_cnn_model(train_dataset, config, device=device)
    lstm_model = train_lstm_model(cnn_model, train_dataset, config, device=device)
    baseline_model = train_baseline_model(train_dataset, config, device=device)
    
    os.makedirs(config["model_dir"], exist_ok=True)
    torch.save(baseline_model.state_dict(), os.path.join(config["model_dir"], "baseline_mlp.pth"))
    torch.save(cnn_model.state_dict(), os.path.join(config["model_dir"], "cnn_drag_predictor.pth"))
    torch.save(lstm_model.state_dict(), os.path.join(config["model_dir"], "lstm_drag.pth"))
    
    logging.info("Training completed. Models saved in the 'models' directory.")
    
if __name__ == "__main__":
    main()
