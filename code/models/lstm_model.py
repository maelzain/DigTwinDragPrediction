import torch
import torch.nn as nn

class LSTMDragPredictor(nn.Module):
    """
    LSTM that takes (seq_len, batch, input_size) latent vectors
    and outputs a drag prediction at each time step.
    """
    def __init__(self, input_size=64, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMDragPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (seq_len, batch, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # shape of lstm_out: (seq_len, batch, hidden_size)
        # pass each time step through a linear layer
        predictions = self.fc(lstm_out)  # (seq_len, batch, output_size)
        return predictions, (h_n, c_n)
