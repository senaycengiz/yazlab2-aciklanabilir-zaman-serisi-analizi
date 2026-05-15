import pandas as pd
import torch

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def create_dataloader(
    csv_path,
    label_column="ATT_FLAG",
    batch_size=32
):

    df = pd.read_csv(csv_path)

    y = df[label_column].astype(float)

    if label_column == "ATT_FLAG":
        y = y.apply(lambda value: 0.0 if value == -999 else 1.0)

    y = y.values

    X = df.drop(columns=[label_column])

    X = X.select_dtypes(include=["number"])

    X_tensor = torch.tensor(
        X.values,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    )

    X_tensor = X_tensor.unsqueeze(-1)

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataloader
