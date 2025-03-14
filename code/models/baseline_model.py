import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    """
    A simple MLP baseline that flattens images
    and directly predicts drag. No LSTM, no fancy
    spatial compression. 
    """
    def __init__(self, input_dim=64*64, hidden_dim=128):
        super(BaselineMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)  # single drag value
        )

    def forward(self, x):
        # x: (batch, 1, 64, 64)
        # flatten to (batch, 64*64)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        out = self.model(x)
        return out
