import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.models.cnn_model import CNNDragPredictor
from code.models.lstm_model import LSTMDragPredictor

def train_cnn_drag_predictor(images, drags, config, device="cpu"):
    """
    Train the CNN model to predict drag from images.
    
    Args:
        images (Tensor): Preprocessed images tensor.
        drags (Tensor): Drag values tensor.
        config (dict): Configuration dictionary containing training parameters.
        device (str): Device to run training on ("cpu" or "cuda").
        
    Returns:
        model (CNNDragPredictor): Trained CNN model.
    """
    # Retrieve CNN training parameters
    epochs = config["cnn"].get("epochs", 150)
    batch_size = config["cnn"].get("batch_size", 256)
    lr = config["cnn"].get("learning_rate", 0.0005)
    latent_dim = config["cnn"].get("latent_dim", 128)
    
    # Prepare dataset and dataloader
    dataset = TensorDataset(images, drags)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the CNN model
    model = CNNDragPredictor(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    logging.info(f"Starting CNN training for {epochs} epochs on {device}")
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for x, drag_true in dataloader:
            x, drag_true = x.to(device), drag_true.to(device)
            optimizer.zero_grad()
            latent, drag_pred = model(x)
            loss_drag = criterion(drag_pred, drag_true)
            loss_drag.backward()
            optimizer.step()
            running_loss += loss_drag.item() * x.size(0)
        epoch_loss = running_loss / len(dataset)
        if epoch % 10 == 0 or epoch == epochs:
            logging.info(f"Epoch [{epoch}/{epochs}] Drag Loss: {epoch_loss:.6f}")
    return model

def train_lstm_drag(cnn_model, images, drags, config, device="cpu"):
    """
    Train the LSTM model to evolve latent features over time for drag prediction.
    The LSTM is trained using latent features extracted from the pre-trained CNN model,
    with an additional physics-informed loss to enforce smoothness.
    
    Args:
        cnn_model (CNNDragPredictor): Pre-trained CNN model for latent extraction.
        images (Tensor): Preprocessed images tensor.
        drags (Tensor): Drag values tensor.
        config (dict): Configuration dictionary containing LSTM training parameters.
        device (str): Device to run training on.
        
    Returns:
        lstm_model (LSTMDragPredictor): Trained LSTM model.
    """
    # Extract latent features from images using the pre-trained CNN
    cnn_model.eval()
    latents = []
    with torch.no_grad():
        for i in range(images.size(0)):
            x_i = images[i:i+1].to(device)
            latent, _ = cnn_model(x_i)
            latents.append(latent.cpu())
    latents = torch.cat(latents, dim=0)  # Shape: (N, latent_dim)
    
    # Prepare dataset and dataloader using latent features
    dataset = TensorDataset(latents, drags)
    batch_size = config["lstm"].get("batch_size", 64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Retrieve LSTM training parameters
    lstm_epochs = config["lstm"].get("epochs", 100)
    lstm_lr = config["lstm"].get("learning_rate", 0.0005)
    physics_loss_weight = config["lstm"].get("physics_loss_weight", 0.005)
    input_size = config["lstm"].get("input_size", 128)
    hidden_size = config["lstm"].get("hidden_size", 64)
    num_layers = config["lstm"].get("num_layers", 2)
    output_size = config["lstm"].get("output_size", 1)
    
    # Initialize the LSTM model
    lstm_model = LSTMDragPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=lstm_lr)
    
    # Reshape latent features for LSTM: (seq_len, batch, input_size)
    latents_all = latents.unsqueeze(1)  # Here, batch=1 for sequence processing
    
    logging.info(f"Training LSTM drag predictor for {lstm_epochs} epochs on {device}")
    lstm_model.train()
    for epoch in range(1, lstm_epochs + 1):
        optimizer.zero_grad()
        pred_seq, _ = lstm_model(latents_all)  # Expected shape: (seq_len, 1, 1)
        pred_seq = pred_seq.squeeze()           # Shape: (seq_len,)
        mse_loss = criterion(pred_seq, drags.squeeze())
        # Compute physics-informed loss (smoothness via finite differences)
        if pred_seq.size(0) > 1:
            diff_pred = pred_seq[1:] - pred_seq[:-1]
            physics_loss = torch.mean(diff_pred ** 2)
        else:
            physics_loss = 0.0
        total_loss = mse_loss + physics_loss_weight * physics_loss
        total_loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == lstm_epochs:
            logging.info(f"Epoch [{epoch}/{lstm_epochs}] LSTM Drag Loss: {total_loss.item():.6f} "
                         f"(MSE: {mse_loss.item():.6f}, Physics: "
                         f"{physics_loss if isinstance(physics_loss, float) else physics_loss.item():.6f})")
    return lstm_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # The following block is a placeholder for integration with your overall training pipeline.
    # In practice, the training functions below are invoked by your main training script.
    
    # Example usage (assuming images and drags tensors are available):
    #
    # from code.utils.data_loader import ReynoldsDataLoader
    # config = load_config("config.yaml")
    # loader = ReynoldsDataLoader(config)
    # data_dict = loader.load_dataset()
    # Combine images and drag data from all valid folders...
    # images_combined = ...
    # drags_combined = ...
    #
    # cnn_model = train_cnn_drag_predictor(images_combined, drags_combined, config, device="cpu")
    # lstm_model = train_lstm_drag(cnn_model, images_combined, drags_combined, config, device="cpu")
    #
    # torch.save(cnn_model.state_dict(), os.path.join("models", "cnn_drag_predictor.pth"))
    # torch.save(lstm_model.state_dict(), os.path.join("models", "lstm_drag.pth"))
    #
    # logging.info("Training completed. Models saved in the 'models' directory.")
