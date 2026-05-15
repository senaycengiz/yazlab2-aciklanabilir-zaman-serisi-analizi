import os
import csv
import torch

from config.config import BATCH_SIZE
from src.data_loader_torch import create_dataloader
from src.models.cnn_model import CNN1DModel
from src.train import train_model
from src.plot_training_history import save_history_csv, save_loss_plot


RESULTS_DIR = "results/cnn"
MODELS_DIR = "models/cnn"
LOGS_DIR = "logs/cnn"
PLOTS_DIR = "results/plots/cnn"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def train_skab_folds():
    fold_results = []

    for fold in range(1, 6):
        print(f"\nSKAB Fold {fold} CNN eğitimi başlatıldı.")

        train_path = f"data/processed/pca/SKAB/fold_{fold}/train.csv"
        test_path = f"data/processed/pca/SKAB/fold_{fold}/test.csv"

        train_loader = create_dataloader(
            train_path,
            label_column="anomaly",
            batch_size=BATCH_SIZE
        )

        test_loader = create_dataloader(
            test_path,
            label_column="anomaly",
            batch_size=BATCH_SIZE
        )

        model = CNN1DModel(input_size=1)

        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=test_loader,
            return_history=True
        )

        model_path = f"{MODELS_DIR}/cnn_skab_fold_{fold}.pt"
        history_csv_path = f"{LOGS_DIR}/cnn_skab_fold_{fold}_loss_history.csv"
        plot_path = f"{PLOTS_DIR}/cnn_skab_fold_{fold}_loss.png"

        torch.save(trained_model.state_dict(), model_path)

        save_history_csv(history, history_csv_path)
        save_loss_plot(
            history,
            plot_path,
            title=f"CNN SKAB Fold {fold} Loss Grafiği"
        )

        fold_results.append({
            "dataset": "SKAB",
            "fold": fold,
            "model": "CNN",
            "model_path": model_path,
            "loss_history_path": history_csv_path,
            "loss_plot_path": plot_path,
            "status": "trained"
        })

    results_path = f"{RESULTS_DIR}/skab_fold_results.csv"

    with open(results_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "fold",
                "model",
                "model_path",
                "loss_history_path",
                "loss_plot_path",
                "status"
            ]
        )
        writer.writeheader()
        writer.writerows(fold_results)

    print(f"\nSKAB fold sonuçları kaydedildi: {results_path}")


def train_batadal():
    batadal_results = []

    datasets = [
        "BATADAL_dataset04"
    ]

    for dataset_name in datasets:
        print(f"\n{dataset_name} CNN eğitimi başlatıldı.")

        base_path = f"data/processed/pca/BATADAL/{dataset_name}"

        train_path = f"{base_path}/train.csv"
        validation_path = f"{base_path}/validation.csv"

        train_loader = create_dataloader(
            train_path,
            batch_size=BATCH_SIZE
        )

        validation_loader = create_dataloader(
            validation_path,
            batch_size=BATCH_SIZE
        )

        model = CNN1DModel(input_size=1)

        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            return_history=True
        )

        model_path = f"{MODELS_DIR}/cnn_{dataset_name}.pt"
        history_csv_path = f"{LOGS_DIR}/cnn_{dataset_name}_loss_history.csv"
        plot_path = f"{PLOTS_DIR}/cnn_{dataset_name}_loss.png"

        torch.save(trained_model.state_dict(), model_path)

        save_history_csv(history, history_csv_path)
        save_loss_plot(
            history,
            plot_path,
            title=f"CNN {dataset_name} Loss Grafiği"
        )

        batadal_results.append({
            "dataset": dataset_name,
            "model": "CNN",
            "model_path": model_path,
            "loss_history_path": history_csv_path,
            "loss_plot_path": plot_path,
            "status": "trained"
        })

    results_path = f"{RESULTS_DIR}/batadal_results.csv"

    with open(results_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "model",
                "model_path",
                "loss_history_path",
                "loss_plot_path",
                "status"
            ]
        )
        writer.writeheader()
        writer.writerows(batadal_results)

    print(f"\nBATADAL sonuçları kaydedildi: {results_path}")


if __name__ == "__main__":
    print("CNN model eğitimleri başlatıldı.")
    train_skab_folds()
    train_batadal()
    print("\nCNN model eğitimleri tamamlandı.")
