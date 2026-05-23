import os
import csv
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.automata_predict import map_unseen_pattern

from config.config import (
    WINDOW_SIZE,
    AUTOMATA_SMOOTHING_ALPHA,
    AUTOMATA_THRESHOLD_QUANTILE,
    UNSEEN_DISTANCE_THRESHOLD
)


RESULTS_DIR = "results/automata_normal_data"
LOGS_DIR = "logs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def normalize_labels(values, label_column):
    if label_column == "ATT_FLAG":
        return [0 if value == -999 else 1 for value in values]

    return [int(float(value)) for value in values]


def create_symbol_labels(raw_labels, symbol_count):
    symbol_labels = []
    total_length = len(raw_labels)

    for index in range(symbol_count):
        start = round(index * total_length / symbol_count)
        end = round((index + 1) * total_length / symbol_count)

        window = raw_labels[start:end]

        if not window:
            symbol_labels.append(0)
        else:
            symbol_labels.append(max(window))

    return symbol_labels


def create_pattern_labels(raw_labels, pattern_count, window_size):
    symbol_count = pattern_count + window_size - 1
    symbol_labels = create_symbol_labels(raw_labels, symbol_count)

    pattern_labels = []

    for index in range(pattern_count):
        window = symbol_labels[index:index + window_size]
        pattern_labels.append(max(window))

    return pattern_labels


def build_transition_probability_model(patterns, smoothing_alpha):
    known_patterns = sorted(set(patterns))

    transition_counts = {}
    outgoing_counts = {}

    for index in range(len(patterns) - 1):
        from_pattern = patterns[index]
        to_pattern = patterns[index + 1]

        key = (from_pattern, to_pattern)

        transition_counts[key] = transition_counts.get(key, 0) + 1
        outgoing_counts[from_pattern] = outgoing_counts.get(from_pattern, 0) + 1

    state_count = len(known_patterns)

    def get_probability(from_pattern, to_pattern):
        count = transition_counts.get((from_pattern, to_pattern), 0)
        outgoing = outgoing_counts.get(from_pattern, 0)

        probability = (
            count + smoothing_alpha
        ) / (
            outgoing + smoothing_alpha * state_count
        )

        return probability

    return known_patterns, get_probability


def calculate_training_threshold(train_patterns, get_probability, quantile):
    probabilities = []

    for index in range(len(train_patterns) - 1):
        from_pattern = train_patterns[index]
        to_pattern = train_patterns[index + 1]

        probability = get_probability(from_pattern, to_pattern)
        probabilities.append(probability)

    if not probabilities:
        return 0.0

    probabilities = sorted(probabilities)
    threshold_index = int(len(probabilities) * quantile)
    threshold_index = min(threshold_index, len(probabilities) - 1)

    return probabilities[threshold_index]


def map_pattern(pattern, known_patterns):
    mapping_result = map_unseen_pattern(
        sequence=pattern,
        known_patterns=known_patterns
    )

    return mapping_result


def evaluate_automata(
    train_pattern_path,
    test_pattern_path,
    label_path,
    label_column,
    dataset,
    fold="-",
    window_size=WINDOW_SIZE
):
    train_df = pd.read_csv(train_pattern_path)
    test_df = pd.read_csv(test_pattern_path)
    label_df = pd.read_csv(label_path)

    if "pattern" not in train_df.columns or "pattern" not in test_df.columns:
        raise ValueError("Pattern dosyalarında 'pattern' kolonu bulunmalıdır.")

    if label_column not in label_df.columns:
        raise ValueError(f"{label_path} dosyasında {label_column} kolonu bulunamadı.")

    train_patterns = train_df["pattern"].astype(str).dropna().tolist()
    test_patterns = test_df["pattern"].astype(str).dropna().tolist()

    raw_labels = normalize_labels(
        label_df[label_column].tolist(),
        label_column
    )

    pattern_labels = create_pattern_labels(
        raw_labels=raw_labels,
        pattern_count=len(test_patterns),
        window_size=window_size
    )

    known_patterns, get_probability = build_transition_probability_model(
        patterns=train_patterns,
        smoothing_alpha=AUTOMATA_SMOOTHING_ALPHA
    )

    probability_threshold = calculate_training_threshold(
        train_patterns=train_patterns,
        get_probability=get_probability,
        quantile=AUTOMATA_THRESHOLD_QUANTILE
    )

    y_true = []
    y_pred = []

    unseen_count = 0
    low_probability_count = 0
    probability_sum = 0.0
    transition_count = 0

    for index in range(1, len(test_patterns)):
        previous_mapping = map_pattern(test_patterns[index - 1], known_patterns)
        current_mapping = map_pattern(test_patterns[index], known_patterns)

        previous_pattern = previous_mapping["mapped_pattern"]
        current_pattern = current_mapping["mapped_pattern"]

        transition_probability = get_probability(
            previous_pattern,
            current_pattern
        )

        is_unseen = (
            previous_mapping["is_unseen"]
            or current_mapping["is_unseen"]
        )

        max_distance = max(
            previous_mapping["distance"],
            current_mapping["distance"]
        )

        is_low_probability = transition_probability < probability_threshold

        prediction = 1 if (
            is_low_probability or
            (is_unseen and max_distance > UNSEEN_DISTANCE_THRESHOLD)
        ) else 0

        if is_unseen:
            unseen_count += 1

        if is_low_probability:
            low_probability_count += 1

        probability_sum += transition_probability
        transition_count += 1

        y_true.append(pattern_labels[index])
        y_pred.append(prediction)

    average_transition_probability = (
        probability_sum / transition_count
        if transition_count > 0
        else 0.0
    )

    return {
        "dataset": dataset,
        "fold": fold,
        "model": "Automata",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "scenario": "normal",
        "window_size": window_size,
        "smoothing_alpha": AUTOMATA_SMOOTHING_ALPHA,
        "threshold_quantile": AUTOMATA_THRESHOLD_QUANTILE,
        "probability_threshold": probability_threshold,
        "train_pattern_count": len(known_patterns),
        "test_pattern_count": len(test_patterns),
        "normal_pattern_count": pattern_labels.count(0),
        "anomaly_pattern_count": pattern_labels.count(1),
        "unseen_transition_count": unseen_count,
        "low_probability_count": low_probability_count,
        "average_transition_probability": average_transition_probability
    }


def evaluate_skab():
    rows = []

    for fold in range(1, 6):
        train_pattern_path = f"data/processed/patterns/SKAB/fold_{fold}/train.csv"
        test_pattern_path = f"data/processed/patterns/SKAB/fold_{fold}/test.csv"
        label_path = f"data/processed/pca/SKAB/fold_{fold}/test.csv"

        print(f"SKAB Fold {fold} - Automata normal veri testi başlatıldı.")

        rows.append(
            evaluate_automata(
                train_pattern_path=train_pattern_path,
                test_pattern_path=test_pattern_path,
                label_path=label_path,
                label_column="anomaly",
                dataset="SKAB",
                fold=fold
            )
        )

    return rows


def evaluate_batadal():
    dataset_name = "BATADAL_dataset04"

    train_pattern_path = f"data/processed/patterns/BATADAL/{dataset_name}/train.csv"
    test_pattern_path = f"data/processed/patterns/BATADAL/{dataset_name}/test.csv"
    label_path = f"data/processed/pca/BATADAL/{dataset_name}/test.csv"

    print(f"{dataset_name} - Automata normal veri testi başlatıldı.")

    return [
        evaluate_automata(
            train_pattern_path=train_pattern_path,
            test_pattern_path=test_pattern_path,
            label_path=label_path,
            label_column="ATT_FLAG",
            dataset=dataset_name,
            fold="-"
        )
    ]


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
        "window_size",
        "smoothing_alpha",
        "threshold_quantile",
        "probability_threshold",
        "train_pattern_count",
        "test_pattern_count",
        "normal_pattern_count",
        "anomaly_pattern_count",
        "unseen_transition_count",
        "low_probability_count",
        "average_transition_probability"
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_skab_summary(rows):
    summary_path = f"{RESULTS_DIR}/skab_automata_normal_summary.csv"

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
            "scenario": "normal"
        })

    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["dataset", "model", "metric", "mean", "std", "scenario"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def write_log(rows):
    log_path = f"{LOGS_DIR}/automata_normal_experiment.log"

    with open(log_path, "w") as file:
        file.write("Gün 23 - Automata Normal Veri Senaryosu Deney Logu\n")
        file.write("=" * 60 + "\n\n")

        for row in rows:
            file.write(f"Dataset: {row['dataset']}\n")
            file.write(f"Fold: {row['fold']}\n")
            file.write(f"Model: {row['model']}\n")
            file.write(f"Scenario: {row['scenario']}\n")
            file.write(f"Window Size: {row['window_size']}\n")
            file.write(f"Smoothing Alpha: {row['smoothing_alpha']}\n")
            file.write(f"Threshold Quantile: {row['threshold_quantile']}\n")
            file.write(f"Probability Threshold: {row['probability_threshold']:.6f}\n")
            file.write(f"Train Pattern Count: {row['train_pattern_count']}\n")
            file.write(f"Test Pattern Count: {row['test_pattern_count']}\n")
            file.write(f"Normal Pattern Count: {row['normal_pattern_count']}\n")
            file.write(f"Anomaly Pattern Count: {row['anomaly_pattern_count']}\n")
            file.write(f"Unseen Transition Count: {row['unseen_transition_count']}\n")
            file.write(f"Low Probability Count: {row['low_probability_count']}\n")
            file.write(
                f"Average Transition Probability: "
                f"{row['average_transition_probability']:.6f}\n"
            )
            file.write(f"Accuracy: {row['accuracy']:.4f}\n")
            file.write(f"Precision: {row['precision']:.4f}\n")
            file.write(f"Recall: {row['recall']:.4f}\n")
            file.write(f"F1-score: {row['f1']:.4f}\n")
            file.write("-" * 60 + "\n")


if __name__ == "__main__":
    print("Gün 23 automata normal veri senaryosu değerlendirmesi başlatıldı.\n")

    skab_rows = evaluate_skab()
    batadal_rows = evaluate_batadal()

    save_csv(
        f"{RESULTS_DIR}/skab_automata_normal_results.csv",
        skab_rows
    )

    save_csv(
        f"{RESULTS_DIR}/batadal_automata_normal_results.csv",
        batadal_rows
    )

    save_skab_summary(skab_rows)
    write_log(skab_rows + batadal_rows)

    print("\nAutomata normal veri senaryosu tamamlandı.")
    print(f"SKAB sonuçları: {RESULTS_DIR}/skab_automata_normal_results.csv")
    print(f"SKAB özet sonuçları: {RESULTS_DIR}/skab_automata_normal_summary.csv")
    print(f"BATADAL sonuçları: {RESULTS_DIR}/batadal_automata_normal_results.csv")
    print(f"Log dosyası: {LOGS_DIR}/automata_normal_experiment.log")
