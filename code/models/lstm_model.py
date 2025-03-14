import torch
import torch.nn as nn

class LSTMDragPredictor(nn.Module):
    """
    LSTM that evolves latent representations to produce time-series drag predictions.
    Input shape: (seq_len, batch, input_size)
    Output shape: (seq_len, batch, output_size)
    """
    def __init__(self, input_size=64, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMDragPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        predictions = self.fc(lstm_out)
        return predictions, (h_n, c_n)
