import logging
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from code.utils.data_loader import ReynoldsDataLoader
from code.models.baseline_model import BaselineMLP
from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info("Config loaded with keys: %s", list(config.keys()))
    return config

def split_dataset_3(images, drags, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Splits the dataset into training, validation, and test sets.
    The ratios must sum to 1.0.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    dataset = TensorDataset(images, drags)
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    logging.info(f"Dataset split into {train_size} training, {val_size} validation, and {test_size} test samples.")
    return train_dataset, val_dataset, test_dataset

def train_baseline_model(train_dataset, config, device="cpu"):
    batch_size = config["baseline"].get("batch_size", 64)
    lr = config["baseline"].get("learning_rate", 1e-3)
    epochs = config["baseline"].get("epochs", 50)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    sample_image, _ = train_dataset[0]
    input_dim = sample_image.numel()
    model = BaselineMLP(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logging.info(f"Training Baseline MLP for {epochs} epochs on {device}...")
    model.train()
    for epoch in range(1, epochs+1):
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
    batch_size = config["cnn"].get("batch_size", 300)
    lr = config["cnn"].get("learning_rate", 1e-3)
    epochs = config["cnn"].get("epochs", 100)
    latent_dim = config["cnn"].get("latent_dim", 64)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = CNNDragPredictor(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logging.info(f"Training CNN for {epochs} epochs on {device}...")
    model.train()
    for epoch in range(1, epochs+1):
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
    Extracts latent representations from the training set using the CNN and trains an LSTM 
    to predict drag over time.
    """
    lstm_epochs = config["lstm"].get("epochs", 50)
    lstm_lr = config["lstm"].get("learning_rate", 1e-3)
    input_size = config["lstm"].get("input_size", 64)
    hidden_size = config["lstm"].get("hidden_size", 32)
    num_layers = config["lstm"].get("num_layers", 1)
    output_size = config["lstm"].get("output_size", 1)

    # Prepare full sequence data from train_dataset
    images_all, drags_all = zip(*train_dataset)
    images_all = torch.stack(images_all).to(device)
    drags_all = torch.stack(drags_all).to(device).squeeze(-1)

    cnn_model.eval()
    with torch.no_grad():
        latents_list = []
        for i in range(images_all.size(0)):
            latent, _ = cnn_model(images_all[i:i+1])
            latents_list.append(latent)
        latents_all = torch.cat(latents_list, dim=0)  # shape: (N, latent_dim)
    
    # Reshape for LSTM: (seq_len, batch, input_size) with batch=1
    latents_all = latents_all.unsqueeze(1)
    
    lstm_model = LSTMDragPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=lstm_lr)

    logging.info(f"Training LSTM for {lstm_epochs} epochs on {device}...")
    lstm_model.train()
    for epoch in range(1, lstm_epochs+1):
        optimizer.zero_grad()
        pred_seq, _ = lstm_model(latents_all)  # (N, 1, 1)
        pred_seq = pred_seq.squeeze()           # (N,)
        loss = criterion(pred_seq, drags_all)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == lstm_epochs:
            logging.info(f"[LSTM] Epoch {epoch}/{lstm_epochs}, Drag Loss: {loss.item():.6f}")
    return lstm_model

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    config = load_config(config_path)
    device = config.get("device", "cpu")

    # Load the dataset from all Reynolds folders
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
    if not all_images:
        logging.error("No valid images found.")
        return

    images_combined = torch.cat(all_images, dim=0)
    drags_combined = torch.cat(all_drags, dim=0)
    logging.info(f"Combined images shape: {images_combined.shape}")
    logging.info(f"Combined drags shape: {drags_combined.shape}")

    # Split the dataset: 70% training, 10% validation, 20% test
    train_dataset, val_dataset, test_dataset = split_dataset_3(images_combined, drags_combined, 0.7, 0.1, 0.2)

    # Train Baseline Model
    baseline_model = train_baseline_model(train_dataset, config, device=device)
    os.makedirs("models", exist_ok=True)
    torch.save(baseline_model.state_dict(), os.path.join("models", "baseline_mlp.pth"))

    # Train CNN Model
    cnn_model = train_cnn_model(train_dataset, config, device=device)
    torch.save(cnn_model.state_dict(), os.path.join("models", "cnn_drag_predictor.pth"))

    # Train LSTM Model using CNN latents
    lstm_model = train_lstm_model(cnn_model, train_dataset, config, device=device)
    torch.save(lstm_model.state_dict(), os.path.join("models", "lstm_drag.pth"))

    # Optionally, you can evaluate on the validation set during training to fine-tune hyperparameters.
    logging.info("Training completed. Models saved in the 'models' directory.")
    
if __name__ == "__main__":
    main()
