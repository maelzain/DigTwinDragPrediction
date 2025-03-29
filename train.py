import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import your model classes (ensure the paths below match your project structure)
from code.models.baseline_model import BaselineMLP
from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor

def train_baseline_model(train_dataset, config, device="cpu"):
    """
    Train the Baseline MLP model using the specified hyperparameters.
    """
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
    """
    Train the CNN model.
    """
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
    Train the LSTM model to evolve CNN latent vectors over time with a physics-informed loss.
    """
    lstm_epochs = config["lstm"].get("epochs", 100)
    lstm_lr = config["lstm"].get("learning_rate", 0.0005)
    physics_loss_weight = config["lstm"].get("physics_loss_weight", 0.005)
    input_size = config["lstm"].get("input_size", 128)
    hidden_size = config["lstm"].get("hidden_size", 64)
    num_layers = config["lstm"].get("num_layers", 1)
    output_size = config["lstm"].get("output_size", 1)
    
    # Extract latent features using the pretrained CNN
    cnn_model.eval()
    latents = []
    with torch.no_grad():
        for i in range(len(train_dataset)):
            x, _ = train_dataset[i]
            latent, _ = cnn_model(x.unsqueeze(0).to(device))
            latents.append(latent.cpu())
    latents = torch.cat(latents, dim=0)  # Shape: (N, latent_dim)
    
    # Prepare dataset for LSTM training
    labels = torch.cat([y for _, y in train_dataset], dim=0)
    dataset = TensorDataset(latents, labels)
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
        pred_seq = pred_seq.squeeze()           # Potentially a 0-dim tensor if only one sample is present
        if pred_seq.dim() == 0:
            pred_seq = pred_seq.unsqueeze(0)
        target = labels.squeeze()
        if target.dim() == 0:
            target = target.unsqueeze(0)
        mse_loss = criterion(pred_seq, target)
        # Physics-informed loss for smooth predictions
        if pred_seq.size(0) > 1:
            diff_pred = pred_seq[1:] - pred_seq[:-1]
            physics_loss = torch.mean(diff_pred ** 2)
        else:
            physics_loss = 0.0
        total_loss = mse_loss + physics_loss_weight * physics_loss
        total_loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == lstm_epochs:
            logging.info(
                f"[LSTM] Epoch {epoch}/{lstm_epochs} - MSE: {mse_loss.item():.6f}, "
                f"Physics: {physics_loss if isinstance(physics_loss, float) else physics_loss.item():.6f}, "
                f"Total: {total_loss.item():.6f}"
            )
    return lstm_model
