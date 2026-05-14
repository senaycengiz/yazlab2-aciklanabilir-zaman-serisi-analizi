import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_PATH = PROJECT_ROOT / "src"
MODELS_PATH = PROJECT_ROOT / "src" / "models"

sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(MODELS_PATH))

from data_loader_torch import create_dataloader
from train import train_model
from cnn_model import CNN1DModel


def test_training_pipeline_with_batadal():

    train_loader = create_dataloader(
        "data/processed/pca/BATADAL/BATADAL_dataset04/train.csv",
        label_column="ATT_FLAG",
        batch_size=16
    )

    validation_loader = create_dataloader(
        "data/processed/pca/BATADAL/BATADAL_dataset04/validation.csv",
        label_column="ATT_FLAG",
        batch_size=16
    )

    model = CNN1DModel(
        input_size=1,
        num_filters=16,
        kernel_size=3,
        output_size=1
    )

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=1,
        learning_rate=0.001,
        device="cpu"
    )

    assert trained_model is not None