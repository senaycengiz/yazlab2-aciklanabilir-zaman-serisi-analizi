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

from config.config import (
    AUTOMATA_SMOOTHING_ALPHA,
    AUTOMATA_THRESHOLD_QUANTILE,
    UNSEEN_DISTANCE_THRESHOLD
)


RESULTS_DIR = "results/parameter_analysis"
LOGS_DIR = "logs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

WINDOW_SIZES = [3, 4, 5, 6]
ALPHABET_SIZES = [3, 4, 5, 6]


def find_pc1_column(df):
    possible_columns = ["PC1", "pc1", "PCA", "pca", "value"]

    for column in possible_columns:
        if column in df.columns:
            return column

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    excluded_columns = ["anomaly", "ATT_FLAG", "label", "target"]

    numeric_columns = [
        column for column in numeric_columns
        if column not in excluded_columns
    ]

    if not numeric_columns:
        raise ValueError("PC1 için kullanılabilecek sayısal kolon bulunamadı.")

    return numeric_columns[0]


def convert_to_sax_symbols(train_values, values, alphabet_size):
    breakpoints = np.quantile(
        train_values,
        np.linspace(0, 1, alphabet_size + 1)[1:-1]
    )

    symbol_indexes = np.digitize(values, breakpoints)

    alphabet = [
        chr(ord("a") + index)
        for index in range(alphabet_size)
    ]

    return [
        alphabet[index]
        for index in symbol_indexes
    ]


def create_patterns(symbols, window_size):
    patterns = []

    for index in range(len(symbols) - window_size + 1):
        pattern = "".join(symbols[index:index + window_size])
        patterns.append(pattern)

    return patterns


def calculate_transition_density(state_count, transition_count):
    if state_count == 0:
        return 0.0

    return transition_count / (state_count ** 2)


def evaluate_parameter_setting(
    train_pca_path,
    test_pca_path,
    label_path,
    label_column,
    dataset,
    fold,
    window_size,
    alphabet_size,
    experiment_type
):
    train_df = pd.read_csv(train_pca_path)
    test_df = pd.read_csv(test_pca_path)
    label_df = pd.read_csv(label_path)

    pc1_column = find_pc1_column(train_df)

    train_values = train_df[pc1_column].dropna().to_numpy()
    test_values = test_df[pc1_column].dropna().to_numpy()

    train_symbols = convert_to_sax_symbols(
        train_values=train_values,
        values=train_values,
        alphabet_size=alphabet_size
    )

    test_symbols = convert_to_sax_symbols(
        train_values=train_values,
        values=test_values,
        alphabet_size=alphabet_size
    )

    train_patterns = create_patterns(
        symbols=train_symbols,
        window_size=window_size
    )

    test_patterns = create_patterns(
        symbols=test_symbols,
        window_size=window_size
    )

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
        "experiment_type": experiment_type,
        "dataset": dataset,
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


def run_all_datasets(window_size, alphabet_size, experiment_type):
    rows = []

    for fold in range(1, 6):
        print(
            f"SKAB Fold {fold} çalışıyor | "
            f"window_size={window_size}, alphabet_size={alphabet_size}"
        )

        rows.append(
            evaluate_parameter_setting(
                train_pca_path=f"data/processed/pca/SKAB/fold_{fold}/train.csv",
                test_pca_path=f"data/processed/pca/SKAB/fold_{fold}/test.csv",
                label_path=f"data/processed/pca/SKAB/fold_{fold}/test.csv",
                label_column="anomaly",
                dataset="SKAB",
                fold=fold,
                window_size=window_size,
                alphabet_size=alphabet_size,
                experiment_type=experiment_type
            )
        )

    dataset_name = "BATADAL_dataset04"

    print(
        f"{dataset_name} çalışıyor | "
        f"window_size={window_size}, alphabet_size={alphabet_size}"
    )

    rows.append(
        evaluate_parameter_setting(
            train_pca_path=f"data/processed/pca/BATADAL/{dataset_name}/train.csv",
            test_pca_path=f"data/processed/pca/BATADAL/{dataset_name}/test.csv",
            label_path=f"data/processed/pca/BATADAL/{dataset_name}/test.csv",
            label_column="ATT_FLAG",
            dataset=dataset_name,
            fold="-",
            window_size=window_size,
            alphabet_size=alphabet_size,
            experiment_type=experiment_type
        )
    )

    return rows


def save_csv(path, rows):
    fieldnames = [
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


def save_summary(path, rows, group_column):
    df = pd.DataFrame(rows)

    summary = df.groupby(
        ["experiment_type", group_column, "dataset"]
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
        transition_density_mean=("transition_density", "mean")
    ).reset_index()

    summary.to_csv(path, index=False)


def write_log(window_rows, alphabet_rows):
    log_path = f"{LOGS_DIR}/parameter_analysis_experiment.log"

    with open(log_path, "w") as file:
        file.write("Parametre Analizi Deney Logu\n")
        file.write("=" * 60 + "\n\n")

        file.write("Window size analizi:\n")
        file.write(
            "- alphabet_size = 3 sabit tutularak "
            "window_size 3, 4, 5, 6 değerleri denenmiştir.\n"
        )
        file.write(
            "- Her deney için performans metrikleri, state sayısı, "
            "transition sayısı ve transition yoğunluğu kaydedilmiştir.\n\n"
        )

        file.write("Alphabet size analizi:\n")
        file.write(
            "- window_size = 4 sabit tutularak "
            "alphabet_size 3, 4, 5, 6 değerleri denenmiştir.\n"
        )
        file.write(
            "- Her deney için performans metrikleri, state sayısı, "
            "transition sayısı ve transition yoğunluğu kaydedilmiştir.\n\n"
        )

        file.write(f"Window size deney satır sayısı: {len(window_rows)}\n")
        file.write(f"Alphabet size deney satır sayısı: {len(alphabet_rows)}\n")


def main():
    print("Parametre analizi başlatıldı.\n")

    window_rows = []

    for window_size in WINDOW_SIZES:
        alphabet_size = 3

        print(
            f"\nWindow size analizi başladı: "
            f"window_size={window_size}, alphabet_size={alphabet_size}"
        )

        window_rows.extend(
            run_all_datasets(
                window_size=window_size,
                alphabet_size=alphabet_size,
                experiment_type="window_size_analysis"
            )
        )

    alphabet_rows = []

    for alphabet_size in ALPHABET_SIZES:
        window_size = 4

        print(
            f"\nAlphabet size analizi başladı: "
            f"window_size={window_size}, alphabet_size={alphabet_size}"
        )

        alphabet_rows.extend(
            run_all_datasets(
                window_size=window_size,
                alphabet_size=alphabet_size,
                experiment_type="alphabet_size_analysis"
            )
        )

    save_csv(
        f"{RESULTS_DIR}/window_size_results.csv",
        window_rows
    )

    save_csv(
        f"{RESULTS_DIR}/alphabet_size_results.csv",
        alphabet_rows
    )

    save_summary(
        f"{RESULTS_DIR}/window_size_summary.csv",
        window_rows,
        group_column="window_size"
    )

    save_summary(
        f"{RESULTS_DIR}/alphabet_size_summary.csv",
        alphabet_rows,
        group_column="alphabet_size"
    )

    write_log(window_rows, alphabet_rows)

    print("\nParametre analizi tamamlandı.")
    print(f"Window size sonuçları: {RESULTS_DIR}/window_size_results.csv")
    print(f"Window size özetleri: {RESULTS_DIR}/window_size_summary.csv")
    print(f"Alphabet size sonuçları: {RESULTS_DIR}/alphabet_size_results.csv")
    print(f"Alphabet size özetleri: {RESULTS_DIR}/alphabet_size_summary.csv")
    print(f"Log dosyası: {LOGS_DIR}/parameter_analysis_experiment.log")


if __name__ == "__main__":
    main()