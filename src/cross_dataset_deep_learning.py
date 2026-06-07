import os
import csv
import torch
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config.config import BATCH_SIZE
from src.data_loader_torch import create_dataloader
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.cnn_model import CNN1DModel


RESULTS_DIR = "results/cross_dataset_deep_learning"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_model(model, model_path, dataloader):
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch).squeeze()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= 0.5).float()

            y_true.extend(y_batch.numpy())
            y_pred.extend(predictions.numpy())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }


def run_skab_to_batadal():
    rows = []

    dataset_name = "BATADAL_dataset04"
    test_path = f"data/processed/pca/BATADAL/{dataset_name}/test.csv"

    test_loader = create_dataloader(
        test_path,
        label_column="ATT_FLAG",
        batch_size=BATCH_SIZE
    )

    model_configs = [
        ("LSTM", LSTMModel, "models/lstm/lstm_skab_fold_{}.pt"),
        ("GRU", GRUModel, "models/gru/gru_skab_fold_{}.pt"),
        ("CNN", CNN1DModel, "models/cnn/cnn_skab_fold_{}.pt"),
    ]

    for fold in range(1, 6):
        for model_name, model_class, model_path_template in model_configs:
            model_path = model_path_template.format(fold)

            print(f"{model_name} | Train: SKAB Fold {fold} -> Test: BATADAL")

            model = model_class(input_size=1)

            metrics = evaluate_model(
                model=model,
                model_path=model_path,
                dataloader=test_loader
            )

            rows.append({
                "model": model_name,
                "source_dataset": "SKAB",
                "target_dataset": "BATADAL_dataset04",
                "fold": fold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"]
            })

    return rows


def run_batadal_to_skab():
    rows = []

    dataset_name = "BATADAL_dataset04"

    model_configs = [
        ("LSTM", LSTMModel, f"models/lstm/lstm_{dataset_name}.pt"),
        ("GRU", GRUModel, f"models/gru/gru_{dataset_name}.pt"),
        ("CNN", CNN1DModel, f"models/cnn/cnn_{dataset_name}.pt"),
    ]

    for fold in range(1, 6):
        test_path = f"data/processed/pca/SKAB/fold_{fold}/test.csv"

        test_loader = create_dataloader(
            test_path,
            label_column="anomaly",
            batch_size=BATCH_SIZE
        )

        for model_name, model_class, model_path in model_configs:
            print(f"{model_name} | Train: BATADAL -> Test: SKAB Fold {fold}")

            model = model_class(input_size=1)

            metrics = evaluate_model(
                model=model,
                model_path=model_path,
                dataloader=test_loader
            )

            rows.append({
                "model": model_name,
                "source_dataset": "BATADAL_dataset04",
                "target_dataset": "SKAB",
                "fold": fold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"]
            })

    return rows


def save_csv(path, rows):
    fieldnames = [
        "model",
        "source_dataset",
        "target_dataset",
        "fold",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(path, rows):
    df = pd.DataFrame(rows)

    summary = df.groupby(
        ["model", "source_dataset", "target_dataset"]
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std")
    ).reset_index()

    summary.to_csv(path, index=False)


def main():
    print("Deep learning cross-dataset testi başlatıldı.\n")

    rows = []
    rows.extend(run_skab_to_batadal())
    rows.extend(run_batadal_to_skab())

    save_csv(
        f"{RESULTS_DIR}/cross_dataset_deep_learning_results.csv",
        rows
    )

    save_summary(
        f"{RESULTS_DIR}/cross_dataset_deep_learning_summary.csv",
        rows
    )

    print("\nDeep learning cross-dataset testi tamamlandı.")
    print(f"Sonuçlar: {RESULTS_DIR}/cross_dataset_deep_learning_results.csv")
    print(f"Özet: {RESULTS_DIR}/cross_dataset_deep_learning_summary.csv")


if __name__ == "__main__":
    main()