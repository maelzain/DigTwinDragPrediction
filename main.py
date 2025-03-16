import os
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from code.utils.data_loader import ReynoldsDataLoader
from code.models.baseline_model import BaselineMLP
from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"Config loaded with keys: {list(config.keys())}")
    return config

def split_dataset(images, drags, train_ratio, test_ratio, eval_ratio):
    """
    Split dataset into training, testing, and evaluation sets.
    """
    total_samples = images.shape[0]
    train_size = int(train_ratio * total_samples)
    test_size = int(test_ratio * total_samples)
    eval_size = total_samples - train_size - test_size
    train_dataset, test_dataset, eval_dataset = random_split(
        TensorDataset(images, drags), [train_size, test_size, eval_size]
    )
    logging.info(f"Dataset split: {train_size} training, {test_size} testing, {eval_size} evaluation samples.")
    return train_dataset, test_dataset, eval_dataset

def train_baseline_model(train_dataset, config, device="cpu"):
    batch_size = config["baseline"].get("batch_size", 64)
    lr = config["baseline"].get("learning_rate", 0.0005)
    epochs = config["baseline"].get("epochs", 100)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    sample_image, _ = train_dataset[0]
    input_dim = sample_image.numel()
    model = BaselineMLP(input_dim=input_dim, hidden_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logging.info(f"Training Baseline MLP for {epochs} epochs on {device}")
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        epoch_loss = running_loss / len(train_dataset)
        if epoch % 10 == 0 or epoch == epochs:
            logging.info(f"[Baseline] Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")
    return model

def train_cnn_model(train_dataset, config, device="cpu"):
    batch_size = config["cnn"].get("batch_size", 256)
    lr = config["cnn"].get("learning_rate", 0.0005)
    epochs = config["cnn"].get("epochs", 150)
    latent_dim = config["cnn"].get("latent_dim", 128)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = CNNDragPredictor(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logging.info(f"Training CNN for {epochs} epochs on {device}")
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            latent, drag_pred = model(x)
            loss = criterion(drag_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        epoch_loss = running_loss / len(train_dataset)
        if epoch % 10 == 0 or epoch == epochs:
            logging.info(f"[CNN] Epoch {epoch}/{epochs}, Drag Loss: {epoch_loss:.6f}")
    return model

def train_lstm_model(cnn_model, train_dataset, config, device="cpu"):
    """
    Trains the LSTM model to evolve CNN latent vectors over time with physics-informed loss.
    """
    lstm_epochs = config["lstm"].get("epochs", 100)
    lstm_lr = config["lstm"].get("learning_rate", 0.0005)
    physics_loss_weight = config["lstm"].get("physics_loss_weight", 0.005)
    input_size = config["lstm"].get("input_size", 128)
    hidden_size = config["lstm"].get("hidden_size", 64)
    num_layers = config["lstm"].get("num_layers", 2)
    output_size = config["lstm"].get("output_size", 1)
    
    # Extract latent features using the pre-trained CNN
    cnn_model.eval()
    latents = []
    with torch.no_grad():
        for i in range(train_dataset.__len__()):
            x, _ = train_dataset[i]
            latent, _ = cnn_model(x.unsqueeze(0).to(device))
            latents.append(latent.cpu())
    latents = torch.cat(latents, dim=0)  # Shape: (N, latent_dim)
    
    # Prepare dataset and dataloader for LSTM training
    dataset = TensorDataset(latents, torch.cat([y for _, y in train_dataset], dim=0))
    batch_size = config["lstm"].get("batch_size", 64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Reshape latent features for LSTM: (seq_len, batch, input_size)
    latents_all = latents.unsqueeze(1)
    
    lstm_model = LSTMDragPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=lstm_lr)
    
    logging.info(f"Training LSTM for {lstm_epochs} epochs on {device}")
    lstm_model.train()
    for epoch in range(1, lstm_epochs + 1):
        optimizer.zero_grad()
        pred_seq, _ = lstm_model(latents_all)  # Expected shape: (seq_len, 1, 1)
        pred_seq = pred_seq.squeeze()           # Shape: (seq_len,)
        mse_loss = criterion(pred_seq, torch.cat([y for _, y in train_dataset], dim=0).squeeze())
        # Physics-informed loss (smoothness via finite differences)
        if pred_seq.size(0) > 1:
            diff_pred = pred_seq[1:] - pred_seq[:-1]
            physics_loss = torch.mean(diff_pred ** 2)
        else:
            physics_loss = 0.0
        total_loss = mse_loss + physics_loss_weight * physics_loss
        total_loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == lstm_epochs:
            logging.info(f"[LSTM] Epoch {epoch}/{lstm_epochs} - MSE: {mse_loss.item():.6f}, Physics: {physics_loss if isinstance(physics_loss, float) else physics_loss.item():.6f}, Total: {total_loss.item():.6f}")
    return lstm_model

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = load_config()
    device = config.get("device", "cpu")
    
    # Load dataset from Reynolds folders
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
    logging.info(f"Combined images shape: {images_combined.shape}")
    logging.info(f"Combined drags shape: {drags_combined.shape}")
    
    # Split dataset: 70% training, 10% testing, 10% evaluation
    train_ratio = config["split"].get("train_ratio", 0.7)
    test_ratio = config["split"].get("test_ratio", 0.1)
    eval_ratio = config["split"].get("eval_ratio", 0.1)
    train_dataset, test_dataset, eval_dataset = split_dataset(images_combined, drags_combined, train_ratio, test_ratio, eval_ratio)
    
    # Train Baseline Model
    baseline_model = train_baseline_model(train_dataset, config, device=device)
    os.makedirs("models", exist_ok=True)
    torch.save(baseline_model.state_dict(), os.path.join("models", "baseline_mlp.pth"))
    
    # Train CNN Model
    cnn_model = train_cnn_model(train_dataset, config, device=device)
    torch.save(cnn_model.state_dict(), os.path.join("models", "cnn_drag_predictor.pth"))
    
    # Train LSTM Model using CNN latent features with physics-informed loss
    lstm_model = train_lstm_model(cnn_model, train_dataset, config, device=device)
    torch.save(lstm_model.state_dict(), os.path.join("models", "lstm_drag.pth"))
    
    logging.info("Training completed. Models saved in the 'models' directory.")

if __name__ == "__main__":
    main()
