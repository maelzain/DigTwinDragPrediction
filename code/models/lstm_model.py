import torch
import torch.nn as nn

class LSTMDragPredictor(nn.Module):
    """
    LSTM-based model to evolve CNN latent vectors over time for drag prediction.
    Expects input shape (seq_len, batch, input_size) and outputs drag predictions.
    
    All hyperparameters must be provided at instantiation.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMDragPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        predictions = self.fc(lstm_out)
        return predictions, (h_n, c_n)
