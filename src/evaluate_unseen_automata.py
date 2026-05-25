import os
import csv
import json
import pandas as pd

from src.automata_predict import (
    load_known_patterns_from_states,
    predict_sequence
)


RESULTS_DIR = "results/unseen"
LOGS_DIR = "logs"
AUTOMATA_STATE_DIR = "data/processed/automata/states"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 1 and int(p) == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 0 and int(p) == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 0 and int(p) == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 1 and int(p) == 0)

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_unseen_file(dataset, fold, train_state_path, unseen_path):
    known_patterns = load_known_patterns_from_states(train_state_path)
    unseen_df = pd.read_csv(unseen_path)

    y_true = []
    y_pred = []
    explanations = []

    for index, row in unseen_df.iterrows():
        pattern = str(row["pattern"])
        original_pattern = str(row.get("original_pattern", pattern))

        details = predict_sequence(
            sequence=pattern,
            known_patterns=known_patterns,
            threshold=1,
            return_details=True
        )

        # Unseen senaryosunda train sözlüğünde görülmeyen pattern anomaly kabul edilir.
        prediction = 1 if details["is_unseen"] else int(details["prediction"])

        # Bu dosyalar özel olarak unseen senaryosu olduğu için gerçek etiket:
        # is_unseen=True ise anomaly(1), değilse normal(0) kabul edilir.
        true_label = 1 if bool(row.get("is_unseen", True)) else 0

        y_true.append(true_label)
        y_pred.append(prediction)

        explanations.append({
            "dataset": dataset,
            "fold": fold,
            "time_step": int(index),
            "state": str(row.get("state", "-")),
            "pattern": pattern,
            "original_pattern": original_pattern,
            "status": "unseen" if details["is_unseen"] else "seen",
            "mapped_to": details["mapped_pattern"],
            "distance": int(details["distance"]),
            "probability": 0.0 if details["is_unseen"] else 1.0,
            "decision": "anomaly" if prediction == 1 else "normal",
            "confidence_score": 1.0 / (1.0 + float(details["distance"]))
        })

    metrics = calculate_metrics(y_true, y_pred)

    result_row = {
        "dataset": dataset,
        "fold": fold,
        "model": "Automata",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "scenario": "unseen",
        "total_rows": len(unseen_df),
        "unseen_rows": int(sum(y_true))
    }

    return result_row, explanations


def save_csv(path, rows):
    fieldnames = [
        "dataset",
        "fold",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "scenario",
        "total_rows",
        "unseen_rows"
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_skab_summary(rows):
    summary_path = f"{RESULTS_DIR}/skab_unseen_automata_summary.csv"

    summary_rows = []

    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = [float(row[metric]) for row in rows]
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
            "model": "Automata",
            "metric": metric,
            "mean": mean_value,
            "std": std_value,
            "scenario": "unseen"
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
    log_path = f"{LOGS_DIR}/unseen_automata_experiment.log"

    with open(log_path, "w") as file:
        file.write("Unseen Veri Senaryosu - Automata Deney Logu\n")
        file.write("=" * 60 + "\n\n")
        file.write(
            "Unseen patternlar train sözlüğünde aranmış, bulunmayan patternlar "
            "Levenshtein mesafesi ile en yakın train patternına eşlenmiştir.\n\n"
        )

        for row in rows:
            file.write(f"Dataset: {row['dataset']}\n")
            file.write(f"Fold: {row['fold']}\n")
            file.write(f"Model: {row['model']}\n")
            file.write(f"Scenario: {row['scenario']}\n")
            file.write(f"Accuracy: {row['accuracy']:.4f}\n")
            file.write(f"Precision: {row['precision']:.4f}\n")
            file.write(f"Recall: {row['recall']:.4f}\n")
            file.write(f"F1-score: {row['f1']:.4f}\n")
            file.write(f"Total Rows: {row['total_rows']}\n")
            file.write(f"Unseen Rows: {row['unseen_rows']}\n")
            file.write("-" * 60 + "\n")


def evaluate_skab():
    rows = []
    explanations = []

    for fold in range(1, 6):
        train_state_path = f"{AUTOMATA_STATE_DIR}/SKAB/fold_{fold}/train.csv"
        unseen_path = f"{RESULTS_DIR}/skab_fold_{fold}_unseen.csv"

        print(f"SKAB Fold {fold} - Automata unseen testi başlatıldı.")

        result_row, fold_explanations = evaluate_unseen_file(
            dataset="SKAB",
            fold=fold,
            train_state_path=train_state_path,
            unseen_path=unseen_path
        )

        rows.append(result_row)
        explanations.extend(fold_explanations)

    return rows, explanations


def evaluate_batadal():
    dataset_name = "BATADAL_dataset04"

    train_state_path = f"{AUTOMATA_STATE_DIR}/BATADAL/{dataset_name}/train.csv"
    unseen_path = f"{RESULTS_DIR}/batadal_unseen.csv"

    print(f"{dataset_name} - Automata unseen testi başlatıldı.")

    result_row, explanations = evaluate_unseen_file(
        dataset=dataset_name,
        fold="-",
        train_state_path=train_state_path,
        unseen_path=unseen_path
    )

    return [result_row], explanations


if __name__ == "__main__":
    print("Unseen veri senaryosu automata değerlendirmesi başlatıldı.\n")

    skab_rows, skab_explanations = evaluate_skab()
    batadal_rows, batadal_explanations = evaluate_batadal()

    save_csv(
        f"{RESULTS_DIR}/skab_unseen_automata_results.csv",
        skab_rows
    )

    save_csv(
        f"{RESULTS_DIR}/batadal_unseen_automata_results.csv",
        batadal_rows
    )

    save_skab_summary(skab_rows)

    all_rows = skab_rows + batadal_rows
    all_explanations = skab_explanations + batadal_explanations

    with open(f"{RESULTS_DIR}/unseen_automata_explanations.json", "w", encoding="utf-8") as file:
        json.dump(all_explanations, file, indent=4, ensure_ascii=False)

    write_log(all_rows)

    print("\nUnseen automata değerlendirmesi tamamlandı.")
    print(f"SKAB sonuçları: {RESULTS_DIR}/skab_unseen_automata_results.csv")
    print(f"SKAB özet sonuçları: {RESULTS_DIR}/skab_unseen_automata_summary.csv")
    print(f"BATADAL sonuçları: {RESULTS_DIR}/batadal_unseen_automata_results.csv")
    print(f"Açıklama JSON: {RESULTS_DIR}/unseen_automata_explanations.json")
    print(f"Log dosyası: {LOGS_DIR}/unseen_automata_experiment.log")
