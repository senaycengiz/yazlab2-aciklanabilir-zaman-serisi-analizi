import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    PCA sonrası tek özellikli zaman serisi verileri için GRU modeli.
    Girdi formatı: (batch_size, sequence_length, input_size)
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_output, _ = self.gru(x)

        last_time_step = gru_output[:, -1, :]

        output = self.fc(last_time_step)

        return output