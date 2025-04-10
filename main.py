import os
import logging
import yaml
import torch
from torch.utils.data import TensorDataset, random_split, Subset, DataLoader
from itertools import product

from code.utils.data_loader import ReynoldsDataLoader
from train import train_baseline_model, train_cnn_lstm_joint_model

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"Configuration loaded with keys: {list(config.keys())}")
    return config

def split_dataset(images, drags, train_ratio, val_ratio, test_ratio):
    """
    Split dataset into training, validation, and test sets.
    """
    total_samples = images.shape[0]
    test_size = int(test_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    train_size = total_samples - test_size - val_size

    full_dataset = TensorDataset(images, drags)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    logging.info(f"Dataset split into Train: {len(train_dataset)} samples, "
                 f"Validation: {len(val_dataset)} samples, Test: {len(test_dataset)} samples.")
    return train_dataset, val_dataset, test_dataset

def evaluate_model_performance(cnn_model, lstm_model, dataset, device):
    """
    Evaluate the performance of a combined CNN+LSTM model on a given dataset.
    """
    cnn_model.eval()
    lstm_model.eval()
    criterion = torch.nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            latent, _ = cnn_model(x)
            latent_seq = latent.unsqueeze(1)
            pred_seq, _ = lstm_model(latent_seq)
            # Use squeeze(1) to remove the sequence dimension
            pred = pred_seq.squeeze(1)
            loss = criterion(pred, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataset)
    return avg_loss

def grid_search_cnn_lstm(train_dataset, config, device, folds=5):
    """
    Perform grid search with k-fold cross-validation for tuning CNN+LSTM hyperparameters.
    
    For each hyperparameter combination the joint training is used.
    """
    # Build parameter grid from config (or use defaults)
    if ("hyperparameter_tuning" in config and
        "cnn_param_grid" in config["hyperparameter_tuning"] and
        "lstm_param_grid" in config["hyperparameter_tuning"]):
        param_grid = {}
        for param, values in config["hyperparameter_tuning"]["cnn_param_grid"].items():
            param_grid[f"cnn.{param}"] = values
        for param, values in config["hyperparameter_tuning"]["lstm_param_grid"].items():
            param_grid[f"lstm.{param}"] = values
    else:
        # Default parameter grid
        param_grid = {
            "cnn.epochs": [30, 50],
            "cnn.latent_dim": [128, 256],
            "lstm.epochs": [50, 100],
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
        # Create a temporary configuration copy for this combination
        temp_config = {section: config[section].copy() if isinstance(config[section], dict) else config[section]
                       for section in config}
        for key, value in zip(keys, combo):
            section, param = key.split('.')
            temp_config[section][param] = value

        # Ensure LSTM input size matches CNN latent dimension
        temp_config["lstm"]["input_size"] = temp_config["cnn"]["latent_dim"]

        cv_losses = []
        for fold in range(folds):
            # Determine indices for the current fold
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold != folds - 1 else total_samples
            val_indices = indices[val_start:val_end]
            train_indices = indices[:val_start] + indices[val_end:]

            cv_train_subset = Subset(train_dataset, train_indices)
            cv_val_subset = Subset(train_dataset, val_indices)

            # Jointly train CNN+LSTM on the current fold
            cnn_model_cv, lstm_model_cv = train_cnn_lstm_joint_model(cv_train_subset, temp_config, device=device)
            # Evaluate on the validation fold
            val_loss = evaluate_model_performance(cnn_model_cv, lstm_model_cv, cv_val_subset, device)
            cv_losses.append(val_loss)

        mean_cv_loss = sum(cv_losses) / len(cv_losses)
        logging.info(f"Grid Search - Params: {combo}, Mean CV Loss: {mean_cv_loss:.6f}")

        if mean_cv_loss < best_score:
            best_score = mean_cv_loss
            best_params = {k: v for k, v in zip(keys, combo)}

    return best_params, best_score

def main():
    """
    Main execution pipeline for training and evaluating CNN+LSTM models.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(42)

    config = load_config()
    device = torch.device(config.get("device", "cpu"))

    # Load dataset using ReynoldsDataLoader
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

    # Split the data: train, validation, and test sets.
    train_dataset, val_dataset, test_dataset = split_dataset(
        images_combined,
        drags_combined,
        config["split"]["train_ratio"],
        config["split"]["eval_ratio"],  # Using eval_ratio for validation
        config["split"]["test_ratio"]
    )

    # Optionally perform grid search for hyperparameter tuning on training data.
    if config.get("grid_search", False):
        logging.info("Starting grid search for CNN+LSTM hyperparameters using K-fold cross-validation on the training data.")
        folds = config.get("robust_evaluation", {}).get("folds", 5)
        best_params, best_score = grid_search_cnn_lstm(train_dataset, config, device=device, folds=folds)
        logging.info(f"Best CNN+LSTM hyperparameters: {best_params} with CV Loss: {best_score:.6f}")

        # Update configuration with the best hyperparameters found.
        for key, value in best_params.items():
            section, param = key.split('.')
            config[section][param] = value

        # Ensure LSTM input size matches the CNN latent dimension.
        config["lstm"]["input_size"] = config["cnn"]["latent_dim"]

    # Train the final models with the selected hyperparameters using joint CNN+LSTM training.
    logging.info("Training final CNN+LSTM model with selected hyperparameters...")
    cnn_model, lstm_model = train_cnn_lstm_joint_model(train_dataset, config, device=device)
    baseline_model = train_baseline_model(train_dataset, config, device=device)

    # Evaluate on the validation set.
    val_loss = evaluate_model_performance(cnn_model, lstm_model, val_dataset, device)
    logging.info(f"Validation loss with best hyperparameters: {val_loss:.6f}")

    # Evaluate on the test set (final performance metric).
    test_loss = evaluate_model_performance(cnn_model, lstm_model, test_dataset, device)
    logging.info(f"Test loss with best hyperparameters: {test_loss:.6f}")

    # Save the trained models.
    os.makedirs(config["model_dir"], exist_ok=True)
    torch.save(baseline_model.state_dict(), os.path.join(config["model_dir"], "baseline_mlp.pth"))
    torch.save(cnn_model.state_dict(), os.path.join(config["model_dir"], "cnn_drag_predictor.pth"))
    torch.save(lstm_model.state_dict(), os.path.join(config["model_dir"], "lstm_drag.pth"))

    logging.info("Training completed. Models saved in the directory: %s", config["model_dir"])

if __name__ == "__main__":
    main()
