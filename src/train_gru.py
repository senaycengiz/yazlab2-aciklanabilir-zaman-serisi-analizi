import os
import csv
import torch

from config.config import BATCH_SIZE
from src.data_loader_torch import create_dataloader
from src.models.gru_model import GRUModel
from src.train import train_model


RESULTS_DIR = "results/gru"
MODELS_DIR = "models/gru"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def train_skab_folds():
    fold_results = []

    for fold in range(1, 6):
        print(f"\nSKAB Fold {fold} GRU eğitimi başlatıldı.")

        train_path = f"data/processed/pca/SKAB/fold_{fold}/train.csv"
        test_path = f"data/processed/pca/SKAB/fold_{fold}/test.csv"

        train_loader = create_dataloader(train_path, label_column="anomaly", batch_size=BATCH_SIZE)
        test_loader = create_dataloader(test_path, label_column="anomaly", batch_size=BATCH_SIZE)

        model = GRUModel(input_size=1)

        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=test_loader
        )

        model_path = f"{MODELS_DIR}/gru_skab_fold_{fold}.pt"
        torch.save(trained_model.state_dict(), model_path)

        fold_results.append({
            "dataset": "SKAB",
            "fold": fold,
            "model_path": model_path,
            "status": "trained"
        })

    results_path = f"{RESULTS_DIR}/skab_fold_results.csv"

    with open(results_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["dataset", "fold", "model_path", "status"]
        )
        writer.writeheader()
        writer.writerows(fold_results)

    print(f"\nSKAB fold sonuçları kaydedildi: {results_path}")


def train_batadal():
    batadal_results = []

    datasets = [
        "BATADAL_dataset03",
        "BATADAL_dataset04"
    ]

    for dataset_name in datasets:
        print(f"\n{dataset_name} GRU eğitimi başlatıldı.")

        base_path = f"data/processed/pca/BATADAL/{dataset_name}"

        train_path = f"{base_path}/train.csv"
        validation_path = f"{base_path}/validation.csv"

        train_loader = create_dataloader(train_path, batch_size=BATCH_SIZE)
        validation_loader = create_dataloader(validation_path, batch_size=BATCH_SIZE)

        model = GRUModel(input_size=1)

        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader
        )

        model_path = f"{MODELS_DIR}/gru_{dataset_name}.pt"
        torch.save(trained_model.state_dict(), model_path)

        batadal_results.append({
            "dataset": dataset_name,
            "model": "GRU",
            "model_path": model_path,
            "status": "trained"
        })

    results_path = f"{RESULTS_DIR}/batadal_results.csv"

    with open(results_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["dataset", "model", "model_path", "status"]
        )
        writer.writeheader()
        writer.writerows(batadal_results)

    print(f"\nBATADAL sonuçları kaydedildi: {results_path}")


if __name__ == "__main__":
    print("GRU model eğitimleri başlatıldı.")
    train_skab_folds()
    train_batadal()
    print("\nGRU model eğitimleri tamamlandı.")
