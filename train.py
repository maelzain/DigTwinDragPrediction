import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_cnn_drag_predictor(images, drags, config, device="cpu"):
    # [Your CNN training code goes here]
    # For example:
    epochs = config["training"].get("epochs", 100)
    batch_size = config["training"].get("batch_size", 64)
    lr = float(config["training"].get("learning_rate", 1e-3))
    latent_dim = config["cnn"].get("latent_dim", 64)
    dataset = TensorDataset(images, drags)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    from code.models.cnn_model import CNNDragPredictor
    model = CNNDragPredictor(latent_dim=latent_dim).to(device)
    criterion_drag = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logging.info(f"Starting CNN training for {epochs} epochs on {device}")
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for x, drag_true in dataloader:
            x = x.to(device)
            drag_true = drag_true.to(device)
            optimizer.zero_grad()
            latent, drag_pred = model(x)
            loss_drag = criterion_drag(drag_pred, drag_true)
            loss = loss_drag
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        epoch_loss = running_loss / len(dataset)
        if epoch % 10 == 0 or epoch == epochs:
            logging.info(f"Epoch [{epoch}/{epochs}] Drag Loss: {epoch_loss:.6f}")
    return model

def train_lstm_drag(cnn_model, images, drags, config, device="cpu"):
    # [Your LSTM training code goes here]
    cnn_model.eval()
    latents = []
    with torch.no_grad():
        for i in range(images.size(0)):
            x_i = images[i:i+1].to(device)
            latent, _ = cnn_model(x_i)
            latents.append(latent.cpu())
    latents = torch.cat(latents, dim=0)  # Shape: (N, latent_dim)
    dataset = TensorDataset(latents, drags)
    batch_size = config["training"].get("batch_size", 64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lstm_epochs = config["lstm"].get("epochs", 50)
    lstm_lr = float(config["lstm"].get("learning_rate", 1e-3))
    from code.models.lstm_model import LSTMDragPredictor
    lstm_model = LSTMDragPredictor(
        input_size=latents.shape[1],
        hidden_size=config["lstm"].get("hidden_size", 32),
        num_layers=config["lstm"].get("num_layers", 1),
        output_size=config["lstm"].get("output_size", 1)
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=lstm_lr)
    logging.info(f"Training LSTM drag predictor for {lstm_epochs} epochs on {device}")
    lstm_model.train()
    for epoch in range(1, lstm_epochs + 1):
        running_loss = 0.0
        for latent_seq, drag_true in dataloader:
            latent_seq = latent_seq.to(device)
            drag_true = drag_true.to(device)
            optimizer.zero_grad()
            # Note: unsqueeze along dim=0 to get shape (1, batch, input_size)
            pred_drag, _ = lstm_model(latent_seq.unsqueeze(0))
            loss = criterion(pred_drag.squeeze(), drag_true.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * latent_seq.size(0)
        epoch_loss = running_loss / len(dataset)
        if epoch % 5 == 0 or epoch == lstm_epochs:
            logging.info(f"Epoch [{epoch}/{lstm_epochs}] LSTM Drag Loss: {epoch_loss:.6f}")
    return lstm_model
