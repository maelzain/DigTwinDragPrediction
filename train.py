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

def train_cnn_model(train_dataset, config, device="cpu"):
    batch_size = config["cnn"]["batch_size"]
    lr = config["cnn"]["learning_rate"]
    epochs = config["cnn"]["epochs"]
    latent_dim = config["cnn"]["latent_dim"]

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
    lstm_epochs = config["lstm"]["epochs"]
    lstm_lr = config["lstm"]["learning_rate"]
    physics_loss_weight = config["lstm"]["physics_loss_weight"]
    input_size = config["lstm"]["input_size"]
    hidden_size = config["lstm"]["hidden_size"]
    num_layers = config["lstm"]["num_layers"]
    output_size = config["lstm"]["output_size"]

    cnn_model.eval()
    latents = []
    with torch.no_grad():
        for i in range(len(train_dataset)):
            x, _ = train_dataset[i]
            latent, _ = cnn_model(x.unsqueeze(0).to(device))
            latents.append(latent.cpu())
    latents = torch.cat(latents, dim=0)
    labels = torch.cat([y for _, y in train_dataset], dim=0)
    dataset = TensorDataset(latents, labels)
    batch_size = config["lstm"]["batch_size"]
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
        pred_seq, _ = lstm_model(latents_all)
        pred_seq = pred_seq.squeeze()
        if pred_seq.dim() == 0:
            pred_seq = pred_seq.unsqueeze(0)
        target = labels.squeeze()
        if target.dim() == 0:
            target = target.unsqueeze(0)
        mse_loss = criterion(pred_seq, target)
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

if __name__ == "__main__":
    # Typically invoked via main.py
    pass
