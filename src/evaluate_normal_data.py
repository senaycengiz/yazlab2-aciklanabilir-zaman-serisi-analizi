import os
import csv
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config.config import BATCH_SIZE
from src.data_loader_torch import create_dataloader
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.cnn_model import CNN1DModel


RESULTS_DIR = "results/normal_data"
LOGS_DIR = "logs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


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


def save_csv(path, rows):
    fieldnames = [
        "dataset",
        "fold",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "scenario"
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_skab_summary(rows):
    summary_path = f"{RESULTS_DIR}/skab_normal_summary.csv"

    models = sorted(set(row["model"] for row in rows))
    summary_rows = []

    for model in models:
        model_rows = [row for row in rows if row["model"] == model]

        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = [float(row[metric]) for row in model_rows]
            mean_value = sum(values) / len(values)

            if len(values) > 1:
                variance = sum(
                    (value - mean_value) ** 2 for value in values
                ) / (len(values) - 1)
                std_value = variance ** 0.5
            else:
                std_value = 0.0

            summary_rows.append({
                "dataset": "SKAB",
                "model": model,
                "metric": metric,
                "mean": mean_value,
                "std": std_value,
                "scenario": "normal"
            })

    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "model",
                "metric",
                "mean",
                "std",
                "scenario"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def write_log(rows):
    log_path = f"{LOGS_DIR}/normal_data_experiment.log"

    with open(log_path, "w") as file:
        file.write(" Normal Veri Senaryosu Deney Logu\n")
        file.write("=" * 50 + "\n\n")

        for row in rows:
            file.write(f"Dataset: {row['dataset']}\n")
            file.write(f"Fold: {row['fold']}\n")
            file.write(f"Model: {row['model']}\n")
            file.write(f"Scenario: {row['scenario']}\n")
            file.write(f"Accuracy: {row['accuracy']:.4f}\n")
            file.write(f"Precision: {row['precision']:.4f}\n")
            file.write(f"Recall: {row['recall']:.4f}\n")
            file.write(f"F1-score: {row['f1']:.4f}\n")
            file.write("-" * 50 + "\n")


def evaluate_skab():
    rows = []

    model_configs = [
        ("LSTM", LSTMModel, "models/lstm/lstm_skab_fold_{}.pt"),
        ("GRU", GRUModel, "models/gru/gru_skab_fold_{}.pt"),
        ("CNN", CNN1DModel, "models/cnn/cnn_skab_fold_{}.pt"),
    ]

    for fold in range(1, 6):
        test_path = f"data/processed/pca/SKAB/fold_{fold}/test.csv"

        test_loader = create_dataloader(
            test_path,
            label_column="anomaly",
            batch_size=BATCH_SIZE
        )

        for model_name, model_class, model_path_template in model_configs:
            model_path = model_path_template.format(fold)

            print(f"SKAB Fold {fold} - {model_name} normal veri testi başlatıldı.")

            model = model_class(input_size=1)

            metrics = evaluate_model(
                model=model,
                model_path=model_path,
                dataloader=test_loader
            )

            rows.append({
                "dataset": "SKAB",
                "fold": fold,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "scenario": "normal"
            })

    return rows


def evaluate_batadal():
    rows = []

    dataset_name = "BATADAL_dataset04"
    test_path = f"data/processed/pca/BATADAL/{dataset_name}/test.csv"

    test_loader = create_dataloader(
        test_path,
        label_column="ATT_FLAG",
        batch_size=BATCH_SIZE
    )

    model_configs = [
        ("LSTM", LSTMModel, f"models/lstm/lstm_{dataset_name}.pt"),
        ("GRU", GRUModel, f"models/gru/gru_{dataset_name}.pt"),
        ("CNN", CNN1DModel, f"models/cnn/cnn_{dataset_name}.pt"),
    ]

    for model_name, model_class, model_path in model_configs:
        print(f"{dataset_name} - {model_name} normal veri testi başlatıldı.")

        model = model_class(input_size=1)

        metrics = evaluate_model(
            model=model,
            model_path=model_path,
            dataloader=test_loader
        )

        rows.append({
            "dataset": dataset_name,
            "fold": "-",
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "scenario": "normal"
        })

    return rows


if __name__ == "__main__":
    print(" Normal veri senaryosu değerlendirmesi başlatıldı.\n")

    skab_rows = evaluate_skab()
    batadal_rows = evaluate_batadal()

    save_csv(
        f"{RESULTS_DIR}/skab_normal_results.csv",
        skab_rows
    )

    save_csv(
        f"{RESULTS_DIR}/batadal_normal_results.csv",
        batadal_rows
    )

    save_skab_summary(skab_rows)
    write_log(skab_rows + batadal_rows)

    print("\nNormal veri senaryosu tamamlandı.")
    print(f"SKAB sonuçları: {RESULTS_DIR}/skab_normal_results.csv")
    print(f"SKAB özet sonuçları: {RESULTS_DIR}/skab_normal_summary.csv")
    print(f"BATADAL sonuçları: {RESULTS_DIR}/batadal_normal_results.csv")
    print(f"Log dosyası: {LOGS_DIR}/normal_data_experiment.log")
