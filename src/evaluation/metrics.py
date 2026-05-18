import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(y_true, y_pred):
    """
    Accuracy, precision, recall ve F1-score hesaplar.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }


def evaluate_prediction_file(input_path, output_path, dataset_name, model_name, fold=None):
    """
    İçinde y_true ve y_pred kolonları olan bir CSV dosyasından metrik hesaplar.
    """
    df = pd.read_csv(input_path)

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("CSV dosyasında y_true ve y_pred kolonları bulunmalıdır.")

    metrics = calculate_metrics(df["y_true"], df["y_pred"])

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "fold": fold,
        **metrics
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame([result]).to_csv(output_path, index=False)

    print(f"Metrikler kaydedildi: {output_path}")
    return result


def calculate_skab_fold_statistics(metrics_files, output_path):
    """
    SKAB fold sonuçlarının ortalamasını ve standart sapmasını hesaplar.
    """
    dfs = [pd.read_csv(file) for file in metrics_files]
    all_metrics = pd.concat(dfs, ignore_index=True)

    metric_columns = ["accuracy", "precision", "recall", "f1_score"]

    mean_values = all_metrics[metric_columns].mean()
    std_values = all_metrics[metric_columns].std()

    summary = pd.DataFrame({
        "metric": metric_columns,
        "mean": mean_values.values,
        "std": std_values.values
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"SKAB fold ortalama/std sonuçları kaydedildi: {output_path}")
    return summary


def save_batadal_time_ordered_results(input_path, output_path):
    """
    BATADAL zaman sıralı test sonuçlarını kaydeder.
    """
    df = pd.read_csv(input_path)

    required_columns = ["y_true", "y_pred"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"{col} kolonu bulunamadı.")

    df = df.reset_index()
    df = df.rename(columns={"index": "time_step"})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"BATADAL zaman sıralı test sonuçları kaydedildi: {output_path}")
    return df