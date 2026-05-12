import torch

from src.models.cnn_model import CNN1DModel


def test_cnn_forward_pass():
    """
    Küçük sahte veri ile 1D-CNN forward pass testi
    """

    batch_size = 16
    sequence_length = 10
    input_size = 1

    model = CNN1DModel(
        input_size=input_size,
        kernel_size=3
    )

    dummy_input = torch.randn(
        batch_size,
        sequence_length,
        input_size
    )

    output = model(dummy_input)

    print("Output shape:", output.shape)

    assert output.shape == (batch_size, 1)
