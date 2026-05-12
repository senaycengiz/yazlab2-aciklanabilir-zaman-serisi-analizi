import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    PCA sonrası tek özellikli zaman serisi verileri için LSTM modeli.
    Girdi formatı: (batch_size, sequence_length, input_size)
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        last_time_step = lstm_output[:, -1, :]
        output = self.fc(last_time_step)

        return output