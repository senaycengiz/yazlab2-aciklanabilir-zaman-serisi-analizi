import os
import csv
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.evaluate_automata_normal_data import (
    evaluate_automata,
    save_csv
)

RESULTS_DIR = "results/automata_noisy_data"
LOGS_DIR = "logs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def create_sax_symbols(values, alphabet_size=3):
    breakpoints = np.quantile(
        values,
        np.linspace(0, 1, alphabet_size + 1)[1:-1]
    )

    symbols = []

    for value in values:
        symbol_index = np.searchsorted(breakpoints, value)
        symbols.append(chr(ord("a") + symbol_index))

    return symbols


def create_patterns(symbols, window_size=4):
    patterns = []

    for i in range(len(symbols) - window_size + 1):
        pattern = "".join(symbols[i:i + window_size])
        patterns.append(pattern)

    return patterns


def prepare_noisy_patterns(noisy_pca_path, output_path):
    df = pd.read_csv(noisy_pca_path)

    if "PC1" not in df.columns:
        raise ValueError(f"{noisy_pca_path} dosyasında PC1 kolonu yok.")

    symbols = create_sax_symbols(df["PC1"].values)
    patterns = create_patterns(symbols)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pd.DataFrame({"pattern": patterns}).to_csv(output_path, index=False)


def evaluate_skab_noisy():
    rows = []

    for fold in range(1, 6):
        noisy_pca_path = f"data/noisy/pca/SKAB/fold_{fold}/test.csv"
        noisy_pattern_path = f"data/noisy/patterns/SKAB/fold_{fold}/test.csv"

        train_pattern_path = f"data/processed/patterns/SKAB/fold_{fold}/train.csv"
        label_path = noisy_pca_path

        print(f"SKAB Fold {fold} - noisy automata pattern hazırlanıyor.")
        prepare_noisy_patterns(noisy_pca_path, noisy_pattern_path)

        print(f"SKAB Fold {fold} - Automata noisy veri testi başlatıldı.")

        row = evaluate_automata(
            train_pattern_path=train_pattern_path,
            test_pattern_path=noisy_pattern_path,
            label_path=label_path,
            label_column="anomaly",
            dataset="SKAB",
            fold=fold
        )

        row["scenario"] = "noisy"
        rows.append(row)

    return rows


def evaluate_batadal_noisy():
    dataset_name = "BATADAL_dataset04"

    noisy_pca_path = f"data/noisy/pca/BATADAL/{dataset_name}/test.csv"
    noisy_pattern_path = f"data/noisy/patterns/BATADAL/{dataset_name}/test.csv"

    train_pattern_path = f"data/processed/patterns/BATADAL/{dataset_name}/train.csv"
    label_path = noisy_pca_path

    print(f"{dataset_name} - noisy automata pattern hazırlanıyor.")
    prepare_noisy_patterns(noisy_pca_path, noisy_pattern_path)

    print(f"{dataset_name} - Automata noisy veri testi başlatıldı.")

    row = evaluate_automata(
        train_pattern_path=train_pattern_path,
        test_pattern_path=noisy_pattern_path,
        label_path=label_path,
        label_column="ATT_FLAG",
        dataset=dataset_name,
        fold="-"
    )

    row["scenario"] = "noisy"

    return [row]


def save_skab_summary(rows):
    summary_path = f"{RESULTS_DIR}/skab_automata_noisy_summary.csv"

    summary_rows = []

    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = [float(row[metric]) for row in rows]
        mean_value = sum(values) / len(values)

        if len(values) > 1:
            variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
            std_value = variance ** 0.5
        else:
            std_value = 0.0

        summary_rows.append({
            "dataset": "SKAB",
            "model": "Automata",
            "metric": metric,
            "mean": mean_value,
            "std": std_value,
            "scenario": "noisy"
        })

    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["dataset", "model", "metric", "mean", "std", "scenario"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def write_log(rows):
    log_path = f"{LOGS_DIR}/automata_noisy_experiment.log"

    with open(log_path, "w") as file:
        file.write("Gün 24 - Automata Gürültülü Veri Senaryosu Deney Logu\n")
        file.write("=" * 60 + "\n\n")

        for row in rows:
            file.write(f"Dataset: {row['dataset']}\n")
            file.write(f"Fold: {row['fold']}\n")
            file.write(f"Scenario: {row['scenario']}\n")
            file.write(f"Accuracy: {row['accuracy']:.4f}\n")
            file.write(f"Precision: {row['precision']:.4f}\n")
            file.write(f"Recall: {row['recall']:.4f}\n")
            file.write(f"F1-score: {row['f1']:.4f}\n")
            file.write(f"Unseen Transition Count: {row['unseen_transition_count']}\n")
            file.write(f"Low Probability Count: {row['low_probability_count']}\n")
            file.write("-" * 60 + "\n")


if __name__ == "__main__":
    print("Gün 24 automata noisy veri senaryosu başlatıldı.\n")

    skab_rows = evaluate_skab_noisy()
    batadal_rows = evaluate_batadal_noisy()

    save_csv(
        f"{RESULTS_DIR}/skab_automata_noisy_results.csv",
        skab_rows
    )

    save_csv(
        f"{RESULTS_DIR}/batadal_automata_noisy_results.csv",
        batadal_rows
    )

    save_skab_summary(skab_rows)
    write_log(skab_rows + batadal_rows)

    print("\nAutomata noisy veri senaryosu tamamlandı.")
    print(f"SKAB sonuçları: {RESULTS_DIR}/skab_automata_noisy_results.csv")
    print(f"SKAB özet sonuçları: {RESULTS_DIR}/skab_automata_noisy_summary.csv")
    print(f"BATADAL sonuçları: {RESULTS_DIR}/batadal_automata_noisy_results.csv")
    print(f"Log dosyası: {LOGS_DIR}/automata_noisy_experiment.log")
