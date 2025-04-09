import os
import logging
import yaml
import torch
from torch.utils.data import TensorDataset, random_split, Subset, DataLoader
from itertools import product

from code.utils.data_loader import ReynoldsDataLoader
from train import train_baseline_model, train_cnn_model, train_lstm_model


def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.

    Parameters:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"Configuration loaded with keys: {list(config.keys())}")
    return config


def split_dataset(images, drags, train_ratio, val_ratio, test_ratio):
    """
    Split dataset into training, validation, and test sets.

    Parameters:
        images (Tensor): Tensor of images.
        drags (Tensor): Tensor of corresponding drag values.
        train_ratio (float): Ratio of training samples.
        val_ratio (float): Ratio of validation samples.
        test_ratio (float): Ratio of test samples.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
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

    Parameters:
        cnn_model: Trained CNN model.
        lstm_model: Trained LSTM model.
        dataset: Dataset for evaluation.
        device: Computation device (CPU or GPU).

    Returns:
        float: Average loss on the dataset.
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
            # Flatten both predictions and targets to ensure matching shapes
            pred = pred_seq.view(-1)
            target = y.view(-1)
            loss = criterion(pred, target)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataset)
    return avg_loss


def grid_search_cnn_lstm(train_dataset, config, device, folds=5):
    """
    Perform grid search with k-fold cross-validation for tuning CNN+LSTM hyperparameters.
    
    This function uses K-fold cross-validation exclusively on the training data. For each
    hyperparameter combination:
      - The training data is split into K folds.
      - In each fold, the model is trained on K-1 folds and evaluated on the remaining fold.
      - The average validation loss across the K folds is computed and used to select the best
        hyperparameter combination.
    
    Parameters:
        train_dataset: The training dataset (only training data is used here).
        config (dict): Configuration dictionary.
        device: Computation device (CPU or GPU).
        folds (int): Number of folds for cross-validation.

    Returns:
        tuple: (best_params, best_score)
            - best_params (dict): Best hyperparameters found.
            - best_score (float): Corresponding average CV loss.
    """
    # Build the parameter grid from config if available; otherwise use defaults.
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
        # Create a temporary copy of the configuration for this hyperparameter combination
        temp_config = {section: config[section].copy() if isinstance(config[section], dict) else config[section]
                       for section in config}
        for key, value in zip(keys, combo):
            section, param = key.split('.')
            temp_config[section][param] = value

        # Ensure LSTM input size matches CNN latent dimension
        temp_config["lstm"]["input_size"] = temp_config["cnn"]["latent_dim"]

        cv_losses = []
        for fold in range(folds):
            # Determine indices for the current fold from the training dataset
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold != folds - 1 else total_samples
            val_indices = indices[val_start:val_end]
            train_indices = indices[:val_start] + indices[val_end:]

            # Create fold-specific subsets (only from training data)
            cv_train_subset = Subset(train_dataset, train_indices)
            cv_val_subset = Subset(train_dataset, val_indices)

            # Train models on the current fold's training subset
            cnn_model = train_cnn_model(cv_train_subset, temp_config, device=device)
            lstm_model = train_lstm_model(cnn_model, cv_train_subset, temp_config, device=device)

            # Evaluate the models on the current fold's validation subset
            val_loss = evaluate_model_performance(cnn_model, lstm_model, cv_val_subset, device)
            cv_losses.append(val_loss)

        # Calculate the mean validation loss for this hyperparameter combination
        mean_cv_loss = sum(cv_losses) / len(cv_losses)
        logging.info(f"Grid Search - Params: {combo}, Mean CV Loss: {mean_cv_loss:.6f}")

        if mean_cv_loss < best_score:
            best_score = mean_cv_loss
            best_params = {k: v for k, v in zip(keys, combo)}

    return best_params, best_score


def main():
    """
    Main execution pipeline for training and evaluating CNN+LSTM models.

    This function:
      - Loads configuration and dataset.
      - Splits the dataset into training, validation, and test sets.
      - Optionally performs grid search for hyperparameter tuning on the training data only.
      - Trains the final models using the best hyperparameters.
      - Evaluates model performance on validation and test sets.
      - Saves the trained models.
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

    # Split the data into training, validation, and test sets.
    # Note: Grid search (hyperparameter tuning) is performed solely on the training data.
    train_dataset, val_dataset, test_dataset = split_dataset(
        images_combined,
        drags_combined,
        config["split"]["train_ratio"],
        config["split"]["eval_ratio"],  # Using eval_ratio as validation ratio
        config["split"]["test_ratio"]
    )

    # Perform grid search for hyperparameter tuning on the training data only, if enabled.
    if config.get("grid_search", False):
        logging.info("Starting grid search for CNN+LSTM hyperparameters using K-fold cross-validation on the training data.")
        folds = config.get("robust_evaluation", {}).get("folds", 5)
        best_params, best_score = grid_search_cnn_lstm(train_dataset, config, device=device, folds=folds)
        logging.info(f"Best CNN+LSTM hyperparameters: {best_params} with CV Loss: {best_score:.6f}")

        # Update configuration with the best hyperparameters found
        for key, value in best_params.items():
            section, param = key.split('.')
            config[section][param] = value

        # Ensure LSTM input size matches the CNN latent dimension
        config["lstm"]["input_size"] = config["cnn"]["latent_dim"]

    # Train final models with the selected hyperparameters on the full training set
    logging.info("Training final models with selected hyperparameters...")
    cnn_model = train_cnn_model(train_dataset, config, device=device)
    lstm_model = train_lstm_model(cnn_model, train_dataset, config, device=device)
    baseline_model = train_baseline_model(train_dataset, config, device=device)

    # Evaluate performance on the validation set
    val_loss = evaluate_model_performance(cnn_model, lstm_model, val_dataset, device)
    logging.info(f"Validation loss with best hyperparameters: {val_loss:.6f}")

    # Evaluate performance on the test set (final performance metric)
    test_loss = evaluate_model_performance(cnn_model, lstm_model, test_dataset, device)
    logging.info(f"Test loss with best hyperparameters: {test_loss:.6f}")

    # Save the trained models
    os.makedirs(config["model_dir"], exist_ok=True)
    torch.save(baseline_model.state_dict(), os.path.join(config["model_dir"], "baseline_mlp.pth"))
    torch.save(cnn_model.state_dict(), os.path.join(config["model_dir"], "cnn_drag_predictor.pth"))
    torch.save(lstm_model.state_dict(), os.path.join(config["model_dir"], "lstm_drag.pth"))

    logging.info("Training completed. Models saved in the directory: %s", config["model_dir"])


if __name__ == "__main__":
    main()
