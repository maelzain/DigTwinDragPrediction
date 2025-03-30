import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    """
    Baseline MLP model for drag prediction.
    Flattens the input (assumed 64x64 grayscale) and predicts a drag value.
    """
    def __init__(self, input_dim=64*64, hidden_dim=128):
        super(BaselineMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        out = self.model(x)
        return out




