import torch
import torch.nn as nn

class CNNDragPredictor(nn.Module):
    def __init__(self, latent_dim=64):
        super(CNNDragPredictor, self).__init__()
        # Spatial Compression (Encoder)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 8 * 8, latent_dim)
        
        # Drag Prediction Head
        self.drag_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        latent = self.fc(x)
        drag_pred = self.drag_head(latent)
        return latent, drag_pred
