import torch

from src.models.lstm_model import LSTMModel


def test_lstm_forward_pass():
    model = LSTMModel(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1
    )

    dummy_input = torch.randn(32, 10, 1)

    output = model(dummy_input)

    print("LSTM çıktı boyutu:", output.shape)

    assert output.shape == (32, 1)


if __name__ == "__main__":
    test_lstm_forward_pass()