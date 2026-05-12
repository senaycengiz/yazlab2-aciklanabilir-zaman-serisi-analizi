import torch

from src.models.gru_model import GRUModel


def test_gru_forward_pass():
    """
    Küçük sahte veri ile GRU forward pass testi
    """

    batch_size = 16
    sequence_length = 10
    input_size = 1

    model = GRUModel(
        input_size=input_size
    )

    dummy_input = torch.randn(
        batch_size,
        sequence_length,
        input_size
    )

    output = model(dummy_input)

    print("Output shape:", output.shape)

    assert output.shape == (batch_size, 1)