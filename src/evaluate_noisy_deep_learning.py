import os
import torch
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.cnn_model import CNN1DModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 10


def create_sequences(values, labels, sequence_length=10):
    X, y = [], []

    for i in range(len(values) - sequence_length):
        X.append(values[i:i + sequence_length])
        y.append(labels[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


def evaluate_model(model, model_path, test_path, label_column):
    df = pd.read_csv(test_path)

    df = df[df[label_column] != -999]

    if len(df) <= SEQUENCE_LENGTH:
        return None

    X, y = create_sequences(
        df["PC1"].values,
        df[label_column].values,
        SEQUENCE_LENGTH
    )

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
        preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0)
    }


def main():
    results = []

    experiments = [
        {
            "dataset": "BATADAL_dataset04",
            "test_path": "data/noisy/pca/BATADAL/BATADAL_dataset04/test.csv",
            "label_column": "ATT_FLAG",
            "models": {
                "LSTM": ("models/lstm/lstm_BATADAL_dataset04.pt", LSTMModel(input_size=1)),
                "GRU": ("models/gru/gru_BATADAL_dataset04.pt", GRUModel(input_size=1)),
                "CNN": ("models/cnn/cnn_BATADAL_dataset04.pt", CNN1DModel(input_size=1)),
            }
        }
    ]

    for fold in range(1, 6):
        experiments.append(
            {
                "dataset": f"SKAB_fold_{fold}",
                "test_path": f"data/noisy/pca/SKAB/fold_{fold}/test.csv",
                "label_column": "anomaly",
                "models": {
                    "LSTM": (f"models/lstm/lstm_skab_fold_{fold}.pt", LSTMModel(input_size=1)),
                    "GRU": (f"models/gru/gru_skab_fold_{fold}.pt", GRUModel(input_size=1)),
                    "CNN": (f"models/cnn/cnn_skab_fold_{fold}.pt", CNN1DModel(input_size=1)),
                }
            }
        )

    for exp in experiments:
        for model_name, (model_path, model) in exp["models"].items():
            if not os.path.exists(model_path):
                print(f"Model bulunamadı: {model_path}")
                continue

            if not os.path.exists(exp["test_path"]):
                print(f"Test dosyası bulunamadı: {exp['test_path']}")
                continue

            metrics = evaluate_model(
                model,
                model_path,
                exp["test_path"],
                exp["label_column"]
            )

            if metrics is None:
                print(f"Yeterli veri yok: {exp['dataset']} - {model_name}")
                continue

            results.append({
                "dataset": exp["dataset"],
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"]
            })

            print(
                f"{exp['dataset']} - {model_name} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"Prec: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

    os.makedirs("results/noise", exist_ok=True)

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        "results/noise/noisy_deep_learning_results.csv",
        index=False
    )

    print("Sonuçlar kaydedildi: results/noise/noisy_deep_learning_results.csv")


if __name__ == "__main__":
    main()
