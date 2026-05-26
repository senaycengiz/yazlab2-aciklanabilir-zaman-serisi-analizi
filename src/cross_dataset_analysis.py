import os
import csv
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.evaluate_automata_normal_data import (
    normalize_labels,
    create_pattern_labels,
    build_transition_probability_model,
    calculate_training_threshold,
    map_pattern
)

from src.parameter_analysis import (
    find_pc1_column,
    convert_to_sax_symbols,
    create_patterns,
    calculate_transition_density
)

from config.config import (
    AUTOMATA_SMOOTHING_ALPHA,
    AUTOMATA_THRESHOLD_QUANTILE,
    UNSEEN_DISTANCE_THRESHOLD
)


RESULTS_DIR = "results/cross_dataset_analysis"
LOGS_DIR = "logs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

WINDOW_SIZE = 4
ALPHABET_SIZE = 3


def load_pc1_values(path):
    df = pd.read_csv(path)
    pc1_column = find_pc1_column(df)
    return df[pc1_column].dropna().to_numpy()


def load_skab_train_values():
    all_values = []

    for fold in range(1, 6):
        path = f"data/processed/pca/SKAB/fold_{fold}/train.csv"
        values = load_pc1_values(path)
        all_values.extend(values)

    return np.array(all_values)


def load_batadal_train_values():
    dataset_name = "BATADAL_dataset04"
    path = f"data/processed/pca/BATADAL/{dataset_name}/train.csv"
    return load_pc1_values(path)


def create_train_patterns(train_values, window_size, alphabet_size):
    train_symbols = convert_to_sax_symbols(
        train_values=train_values,
        values=train_values,
        alphabet_size=alphabet_size
    )

    return create_patterns(
        symbols=train_symbols,
        window_size=window_size
    )


def create_test_patterns(source_train_values, target_test_values, window_size, alphabet_size):
    test_symbols = convert_to_sax_symbols(
        train_values=source_train_values,
        values=target_test_values,
        alphabet_size=alphabet_size
    )

    return create_patterns(
        symbols=test_symbols,
        window_size=window_size
    )


def evaluate_cross_dataset(
    source_dataset,
    target_dataset,
    source_train_values,
    target_test_path,
    target_label_path,
    target_label_column,
    fold,
    window_size=WINDOW_SIZE,
    alphabet_size=ALPHABET_SIZE
):
    target_test_values = load_pc1_values(target_test_path)

    train_patterns = create_train_patterns(
        train_values=source_train_values,
        window_size=window_size,
        alphabet_size=alphabet_size
    )

    test_patterns = create_test_patterns(
        source_train_values=source_train_values,
        target_test_values=target_test_values,
        window_size=window_size,
        alphabet_size=alphabet_size
    )

    label_df = pd.read_csv(target_label_path)

    raw_labels = normalize_labels(
        label_df[target_label_column].tolist(),
        target_label_column
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

    transition_set = set()

    for index in range(len(train_patterns) - 1):
        transition_set.add((train_patterns[index], train_patterns[index + 1]))

    state_count = len(known_patterns)
    transition_count = len(transition_set)
    transition_density = calculate_transition_density(
        state_count=state_count,
        transition_count=transition_count
    )

    y_true = []
    y_pred = []

    unseen_count = 0
    low_probability_count = 0
    probability_sum = 0.0
    evaluated_transition_count = 0

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
            is_low_probability
            or (is_unseen and max_distance > UNSEEN_DISTANCE_THRESHOLD)
        ) else 0

        if is_unseen:
            unseen_count += 1

        if is_low_probability:
            low_probability_count += 1

        probability_sum += transition_probability
        evaluated_transition_count += 1

        y_true.append(pattern_labels[index])
        y_pred.append(prediction)

    average_transition_probability = (
        probability_sum / evaluated_transition_count
        if evaluated_transition_count > 0
        else 0.0
    )

    return {
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "fold": fold,
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "state_count": state_count,
        "transition_count": transition_count,
        "transition_density": transition_density,
        "train_pattern_count": len(train_patterns),
        "test_pattern_count": len(test_patterns),
        "probability_threshold": probability_threshold,
        "unseen_transition_count": unseen_count,
        "low_probability_count": low_probability_count,
        "average_transition_probability": average_transition_probability
    }


def run_skab_to_batadal():
    rows = []

    source_train_values = load_skab_train_values()
    dataset_name = "BATADAL_dataset04"

    print("SKAB -> BATADAL cross dataset testi başlatıldı.")

    rows.append(
        evaluate_cross_dataset(
            source_dataset="SKAB",
            target_dataset=dataset_name,
            source_train_values=source_train_values,
            target_test_path=f"data/processed/pca/BATADAL/{dataset_name}/test.csv",
            target_label_path=f"data/processed/pca/BATADAL/{dataset_name}/test.csv",
            target_label_column="ATT_FLAG",
            fold="-"
        )
    )

    return rows


def run_batadal_to_skab():
    rows = []

    source_train_values = load_batadal_train_values()

    for fold in range(1, 6):
        print(f"BATADAL -> SKAB Fold {fold} cross dataset testi başlatıldı.")

        rows.append(
            evaluate_cross_dataset(
                source_dataset="BATADAL_dataset04",
                target_dataset="SKAB",
                source_train_values=source_train_values,
                target_test_path=f"data/processed/pca/SKAB/fold_{fold}/test.csv",
                target_label_path=f"data/processed/pca/SKAB/fold_{fold}/test.csv",
                target_label_column="anomaly",
                fold=fold
            )
        )

    return rows


def save_csv(path, rows):
    fieldnames = [
        "source_dataset",
        "target_dataset",
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


def save_summary(path, rows):
    df = pd.DataFrame(rows)

    summary = df.groupby(
        ["source_dataset", "target_dataset"]
    ).agg(
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
        transition_density_mean=("transition_density", "mean"),
        unseen_transition_count_mean=("unseen_transition_count", "mean"),
        low_probability_count_mean=("low_probability_count", "mean")
    ).reset_index()

    summary.to_csv(path, index=False)


def write_log(rows):
    log_path = f"{LOGS_DIR}/cross_dataset_analysis_experiment.log"

    with open(log_path, "w") as file:
        file.write("Cross Dataset Analizi Deney Logu\n")
        file.write("=" * 60 + "\n\n")

        file.write("Deney açıklaması:\n")
        file.write("- PCA sonrası PC1 temsili ortak temsil olarak kullanılmıştır.\n")
        file.write("- Bir veri setinde öğrenilen automata yapısı diğer veri setinin test verisi üzerinde denenmiştir.\n")
        file.write("- SKAB -> BATADAL ve BATADAL -> SKAB yönleri ayrı ayrı değerlendirilmiştir.\n")
        file.write("- Performans metrikleri, state sayısı, transition sayısı ve transition yoğunluğu kaydedilmiştir.\n\n")

        file.write(f"Toplam deney satır sayısı: {len(rows)}\n")


def main():
    print("Cross dataset analizi başlatıldı.\n")

    rows = []
    rows.extend(run_skab_to_batadal())
    rows.extend(run_batadal_to_skab())

    save_csv(
        f"{RESULTS_DIR}/cross_dataset_results.csv",
        rows
    )

    save_summary(
        f"{RESULTS_DIR}/cross_dataset_summary.csv",
        rows
    )

    write_log(rows)

    print("\nCross dataset analizi tamamlandı.")
    print(f"Sonuç dosyası: {RESULTS_DIR}/cross_dataset_results.csv")
    print(f"Özet dosyası: {RESULTS_DIR}/cross_dataset_summary.csv")
    print(f"Log dosyası: {LOGS_DIR}/cross_dataset_analysis_experiment.log")


if __name__ == "__main__":
    main()