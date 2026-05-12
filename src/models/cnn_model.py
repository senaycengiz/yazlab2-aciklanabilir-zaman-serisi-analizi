import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    """
    PCA sonrası tek özellikli zaman serisi verileri için 1D-CNN modeli.
    Girdi formatı: (batch_size, sequence_length, input_size)

    Conv1D katmanı PyTorch'ta (batch_size, input_size, sequence_length)
    formatı beklediği için forward içinde transpose işlemi yapılır.
    """

    def __init__(self, input_size=1, num_filters=32, kernel_size=3, output_size=1):
        super(CNN1DModel, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(num_filters, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.squeeze(-1)

        output = self.fc(x)

        return output
