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

def split_dataset(images, drags, split_ratio=0.8):
    """
    Splits the combined dataset into train and test sets.
    """
    dataset = TensorDataset(images, drags)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def train_baseline_model(train_dataset, config, device="cpu"):
    """
    Trains a simple MLP baseline on flattened images.
    """
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
            x = x.to(device)
            y = y.to(device)
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
    """
    Trains a CNN that outputs latent vectors and direct drag predictions.
    """
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
            x = x.to(device)
            y = y.to(device)
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
    Uses the CNN to extract latent vectors, then trains an LSTM to evolve
    them over time to predict drag at each step.
    """
    # Step 1: Convert the entire train_dataset into (latent_seq, drag_seq).
    # We'll assume each folder has a time sequence. 
    # For simplicity, treat the entire train set as one long sequence. 
    # Alternatively, you can do folder-wise sequences.
    lstm_epochs = config["lstm"].get("epochs", 50)
    lstm_lr = config["lstm"].get("learning_rate", 1e-3)
    input_size = config["lstm"].get("input_size", 64)
    hidden_size = config["lstm"].get("hidden_size", 32)
    num_layers = config["lstm"].get("num_layers", 1)
    output_size = config["lstm"].get("output_size", 1)
    batch_size = 1  # We'll feed entire sequences at once for simplicity.

    # Prepare latents & drags
    images_all, drags_all = zip(*train_dataset)  # each is a tuple of (tensor, tensor)
    images_all = torch.stack(images_all)  # shape: (N, 1, 64, 64)
    drags_all = torch.stack(drags_all)    # shape: (N, 1)
    images_all = images_all.to(device)
    drags_all = drags_all.to(device)

    cnn_model.eval()
    with torch.no_grad():
        latents_list = []
        for i in range(images_all.size(0)):
            latent, _ = cnn_model(images_all[i:i+1])
            latents_list.append(latent)
        latents_all = torch.cat(latents_list, dim=0)  # shape: (N, latent_dim)

    # We want a shape (seq_len, batch, input_size) for LSTM. 
    # Here, batch=1, seq_len=N, input_size=latent_dim
    latents_all = latents_all.unsqueeze(1)  # (N, 1, latent_dim)
    drags_all = drags_all.squeeze(-1)       # (N,)

    # Create LSTM model
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
        pred_seq, _ = lstm_model(latents_all)  # shape: (N, 1, 1)
        pred_seq = pred_seq.squeeze()          # (N,)
        loss = criterion(pred_seq, drags_all)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == lstm_epochs:
            logging.info(f"[LSTM] Epoch {epoch}/{lstm_epochs}, Drag Loss: {loss.item():.6f}")

    return lstm_model

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    config = load_config(config_path)
    device = config.get("device", "cpu")

    # Load dataset
    loader = ReynoldsDataLoader(config)
    dataset = loader.load_dataset()

    # Combine all images and drags from all Reynolds folders
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

    # Split dataset into train/test
    train_dataset, test_dataset = split_dataset(images_combined, drags_combined, split_ratio=0.8)

    # 1) Baseline model
    baseline_model = train_baseline_model(train_dataset, config, device=device)
    os.makedirs("models", exist_ok=True)
    torch.save(baseline_model.state_dict(), os.path.join("models", "baseline_mlp.pth"))

    # 2) CNN model
    cnn_model = train_cnn_model(train_dataset, config, device=device)
    torch.save(cnn_model.state_dict(), os.path.join("models", "cnn_drag_predictor.pth"))

    # 3) LSTM model
    lstm_model = train_lstm_model(cnn_model, train_dataset, config, device=device)
    torch.save(lstm_model.state_dict(), os.path.join("models", "lstm_drag.pth"))

if __name__ == "__main__":
    main()
