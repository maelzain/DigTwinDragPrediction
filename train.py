import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.models.baseline_model import BaselineMLP
from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor

def train_baseline_model(train_dataset, config, device="cpu"):
    batch_size = config["baseline"]["batch_size"]
    lr = config["baseline"]["learning_rate"]
    epochs = config["baseline"]["epochs"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Compute input_dim from config["resize"]
    resize_dims = config["resize"]
    input_dim = resize_dims[0] * resize_dims[1]
    hidden_dim = config["baseline"]["hidden_dim"]
    model = BaselineMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
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

def train_cnn_lstm_joint_model(train_dataset, config, device="cpu"):
    """
    Train the CNN and LSTM models jointly in one forward pass.
    
    This integrated pipeline passes each input batch through the CNN to produce
    latent features, then unsqueezes the latent representation into a single-time-step
    sequence that is fed to the LSTM. Both networks are updated together.
    """
    # Hyperparameters for joint training:
    joint_epochs = config["lstm"]["epochs"]  # You may adjust or add a new parameter (e.g. "joint_epochs")
    batch_size = config["cnn"]["batch_size"]   # Using the CNN batch size; you can add a joint-specific one if desired.
    learning_rate = config["lstm"]["learning_rate"]
    physics_loss_weight = config["lstm"].get("physics_loss_weight", 0.0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize CNN model
    latent_dim = config["cnn"]["latent_dim"]
    cnn_model = CNNDragPredictor(latent_dim=latent_dim).to(device)
    # Ensure the LSTM input size matches the CNN latent dimension
    lstm_input_size = latent_dim
    lstm_model = LSTMDragPredictor(
        input_size=lstm_input_size,
        hidden_size=config["lstm"]["hidden_size"],
        num_layers=config["lstm"]["num_layers"],
        output_size=config["lstm"]["output_size"]
    ).to(device)

    # Combine parameters for joint optimization
    optimizer = optim.Adam(list(cnn_model.parameters()) + list(lstm_model.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()

    logging.info(f"Training joint CNN+LSTM model for {joint_epochs} epochs on {device}")
    cnn_model.train()
    lstm_model.train()

    for epoch in range(1, joint_epochs + 1):
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # Forward pass through the CNN
            latent, _ = cnn_model(x)
            # Add a “sequence” dimension – each image is treated as a one-time-step sample.
            latent_seq = latent.unsqueeze(1)  # Shape: (batch, 1, latent_dim)
            # Forward pass through the LSTM
            pred_seq, _ = lstm_model(latent_seq)  # Expected shape: (batch, 1, output_size)
            # Remove the sequence dimension to obtain predictions with shape: (batch, output_size)
            pred = pred_seq.squeeze(1)
            loss = criterion(pred, y)
            
            # Optionally add a physics-informed loss term if sequence length is greater than 1.
            if pred_seq.size(1) > 1:
                diff_pred = pred_seq[:, 1:, :] - pred_seq[:, :-1, :]
                physics_loss = torch.mean(diff_pred ** 2)
            else:
                physics_loss = 0.0

            total_loss = loss + physics_loss_weight * physics_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * x.size(0)
        epoch_loss = running_loss / len(train_dataset)
        if epoch % 5 == 0 or epoch == joint_epochs:
            logging.info(f"[Joint CNN+LSTM] Epoch {epoch}/{joint_epochs}, Loss: {epoch_loss:.6f}")
    return cnn_model, lstm_model

if __name__ == "__main__":
    # This script is typically imported by main.py, so this block is not used.
    pass
