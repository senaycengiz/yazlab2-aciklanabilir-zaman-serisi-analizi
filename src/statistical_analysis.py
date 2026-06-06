import os
import csv
import random
import pandas as pd
import numpy as np
from config.config import WINDOW_SIZE, ALPHABET_SIZE
from scipy.stats import wilcoxon

SCIPY_AVAILABLE = True

from src.parameter_analysis import evaluate_parameter_setting


RESULTS_DIR = "results/statistical_analysis"
LOGS_DIR = "logs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

SEEDS = [42, 123, 2026, 7, 999]




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_skab_seed_experiments(seed):
    rows = []

    for fold in range(1, 6):
        print(f"SKAB Fold {fold} çalışıyor | seed={seed}")

        row = evaluate_parameter_setting(
            train_pca_path=f"data/processed/pca/SKAB/fold_{fold}/train.csv",
            test_pca_path=f"data/processed/pca/SKAB/fold_{fold}/test.csv",
            label_path=f"data/processed/pca/SKAB/fold_{fold}/test.csv",
            label_column="anomaly",
            dataset="SKAB",
            fold=fold,
            window_size=WINDOW_SIZE,
            alphabet_size=ALPHABET_SIZE,
            experiment_type="seed_analysis"
        )

        row["seed"] = seed
        rows.append(row)

    return rows


def run_batadal_seed_experiments(seed):
    dataset_name = "BATADAL_dataset04"

    print(f"{dataset_name} çalışıyor | seed={seed}")

    row = evaluate_parameter_setting(
        train_pca_path=f"data/processed/pca/BATADAL/{dataset_name}/train.csv",
        test_pca_path=f"data/processed/pca/BATADAL/{dataset_name}/test.csv",
        label_path=f"data/processed/pca/BATADAL/{dataset_name}/test.csv",
        label_column="ATT_FLAG",
        dataset=dataset_name,
        fold="-",
        window_size=WINDOW_SIZE,
        alphabet_size=ALPHABET_SIZE,
        experiment_type="seed_analysis"
    )

    row["seed"] = seed

    return [row]


def save_csv(path, rows):
    fieldnames = [
        "seed",
        "experiment_type",
        "dataset",
        "fold",
        "window_size",
        "alphabet_size",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "state_count",
        "transition_count",
        "transition_density",
        "train_pattern_count",
        "test_pattern_count",
        "probability_threshold",
        "unseen_transition_count",
        "low_probability_count",
        "average_transition_probability"
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_skab_fold_summary(rows):
    df = pd.DataFrame(rows)
    skab_df = df[df["dataset"] == "SKAB"]

    summary = skab_df.groupby(["dataset", "fold"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        state_count_mean=("state_count", "mean"),
        transition_count_mean=("transition_count", "mean"),
        transition_density_mean=("transition_density", "mean")
    ).reset_index()

    summary.to_csv(
        f"{RESULTS_DIR}/skab_fold_seed_summary.csv",
        index=False
    )


def save_batadal_seed_summary(rows):
    df = pd.DataFrame(rows)
    batadal_df = df[df["dataset"] == "BATADAL_dataset04"]

    summary_rows = []

    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = batadal_df[metric].astype(float)

        summary_rows.append({
            "dataset": "BATADAL_dataset04",
            "metric": metric,
            "mean": values.mean(),
            "std": values.std()
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        f"{RESULTS_DIR}/batadal_seed_summary.csv",
        index=False
    )


def save_overall_summary(rows):
    df = pd.DataFrame(rows)

    summary = df.groupby(["dataset"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        state_count_mean=("state_count", "mean"),
        transition_count_mean=("transition_count", "mean"),
        transition_density_mean=("transition_density", "mean")
    ).reset_index()

    summary.to_csv(
        f"{RESULTS_DIR}/overall_seed_summary.csv",
        index=False
    )


def save_batadal_time_ordered_results(rows):
    batadal_rows = [
        row for row in rows
        if row["dataset"] == "BATADAL_dataset04"
    ]

    save_csv(
        f"{RESULTS_DIR}/batadal_time_ordered_seed_results.csv",
        batadal_rows
    )


def run_wilcoxon_test(rows):
    df = pd.DataFrame(rows)

    skab_df = df[df["dataset"] == "SKAB"]

    fold_mean_df = skab_df.groupby("fold").agg(
        accuracy_mean=("accuracy", "mean"),
        f1_mean=("f1", "mean")
    ).reset_index()

    accuracy_values = fold_mean_df["accuracy_mean"].tolist()
    f1_values = fold_mean_df["f1_mean"].tolist()

    test_rows = []

    if SCIPY_AVAILABLE and len(accuracy_values) == len(f1_values):
        statistic, p_value = wilcoxon(accuracy_values, f1_values)

        test_rows.append({
            "test_name": "Wilcoxon",
            "comparison": "SKAB fold accuracy mean vs SKAB fold f1 mean",
            "statistic": statistic,
            "p_value": p_value,
            "interpretation": (
                "p_value < 0.05 ise iki metrik dağılımı arasında "
                "istatistiksel olarak anlamlı fark olduğu kabul edilir."
            )
        })
    else:
        test_rows.append({
            "test_name": "Wilcoxon",
            "comparison": "SKAB fold accuracy mean vs SKAB fold f1 mean",
            "statistic": "not_available",
            "p_value": "not_available",
            "interpretation": (
                "scipy kurulumu bulunamadığı için Wilcoxon testi çalıştırılamadı."
            )
        })

    pd.DataFrame(test_rows).to_csv(
        f"{RESULTS_DIR}/statistical_significance_results.csv",
        index=False
    )


def write_log(rows):
    log_path = f"{LOGS_DIR}/statistical_analysis_experiment.log"

    with open(log_path, "w") as file:
        file.write("İstatistiksel Analiz Deney Logu\n")
        file.write("=" * 60 + "\n\n")

        file.write("Kullanılan seed değerleri:\n")
        file.write(", ".join(str(seed) for seed in SEEDS))
        file.write("\n\n")

        file.write("Deney açıklaması:\n")
        file.write("- Automata deneyleri farklı seed değerleriyle tekrar çalıştırılmıştır.\n")
        file.write("- SKAB için fold bazlı ortalama ve standart sapma hesaplanmıştır.\n")
        file.write("- BATADAL için seed bazlı ortalama ve standart sapma hesaplanmıştır.\n")
        file.write("- BATADAL zaman sıralı test sonuçları ayrı dosyada raporlanmıştır.\n")
        file.write("- SKAB fold sonuçları üzerinde Wilcoxon testi uygulanmıştır.\n\n")

        file.write(f"Toplam sonuç satırı: {len(rows)}\n")


def main():
    print("İstatistiksel analiz başlatıldı.\n")

    all_rows = []

    for seed in SEEDS:
        print(f"\nSeed deneyleri başladı: seed={seed}")

        set_seed(seed)

        all_rows.extend(run_skab_seed_experiments(seed))
        all_rows.extend(run_batadal_seed_experiments(seed))

    save_csv(
        f"{RESULTS_DIR}/seed_experiment_results.csv",
        all_rows
    )

    save_skab_fold_summary(all_rows)
    save_batadal_seed_summary(all_rows)
    save_overall_summary(all_rows)
    save_batadal_time_ordered_results(all_rows)
    run_wilcoxon_test(all_rows)
    write_log(all_rows)

    print("\nİstatistiksel analiz tamamlandı.")
    print(f"Seed deney sonuçları: {RESULTS_DIR}/seed_experiment_results.csv")
    print(f"SKAB fold özeti: {RESULTS_DIR}/skab_fold_seed_summary.csv")
    print(f"BATADAL seed özeti: {RESULTS_DIR}/batadal_seed_summary.csv")
    print(f"Genel özet: {RESULTS_DIR}/overall_seed_summary.csv")
    print(f"BATADAL zaman sıralı sonuçları: {RESULTS_DIR}/batadal_time_ordered_seed_results.csv")
    print(f"İstatistiksel anlamlılık testi: {RESULTS_DIR}/statistical_significance_results.csv")
    print(f"Log dosyası: {LOGS_DIR}/statistical_analysis_experiment.log")


if __name__ == "__main__":
    main()