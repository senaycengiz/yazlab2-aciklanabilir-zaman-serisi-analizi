from pathlib import Path

import pandas as pd

from config.config import (
    WINDOW_SIZE,
    AUTOMATA_SMOOTHING_ALPHA,
    AUTOMATA_THRESHOLD_QUANTILE,
    UNSEEN_DISTANCE_THRESHOLD
)

from src.automata_predict import map_unseen_pattern
from src.evaluate_automata_normal_data import (
    normalize_labels,
    create_pattern_labels,
    build_transition_probability_model,
    calculate_training_threshold
)


OUTPUT_DIR = Path("results/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_prediction_rows(
    train_pattern_path,
    test_pattern_path,
    label_path,
    label_column,
    dataset,
    fold="-",
    scenario="normal",
    window_size=WINDOW_SIZE
):
    train_df = pd.read_csv(train_pattern_path)
    test_df = pd.read_csv(test_pattern_path)
    label_df = pd.read_csv(label_path)

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

    prediction_rows = []

    for index in range(1, len(test_patterns)):
        previous_pattern_raw = test_patterns[index - 1]
        current_pattern_raw = test_patterns[index]

        previous_mapping = map_unseen_pattern(
            sequence=previous_pattern_raw,
            known_patterns=known_patterns
        )

        current_mapping = map_unseen_pattern(
            sequence=current_pattern_raw,
            known_patterns=known_patterns
        )

        previous_pattern = previous_mapping["mapped_pattern"]
        current_pattern = current_mapping["mapped_pattern"]

        transition_probability = get_probability(
            previous_pattern,
            current_pattern
        )

        path_probability = transition_probability

        is_unseen = (
            previous_mapping["is_unseen"]
            or current_mapping["is_unseen"]
        )

        max_distance = max(
            previous_mapping["distance"],
            current_mapping["distance"]
        )

        is_low_probability = transition_probability < probability_threshold

        y_pred = 1 if (
            is_low_probability or
            (is_unseen and max_distance > UNSEEN_DISTANCE_THRESHOLD)
        ) else 0

        y_true = int(pattern_labels[index])

        anomaly_score = 1 - path_probability

        prediction_rows.append({
            "dataset": dataset,
            "fold": fold,
            "scenario": scenario,
            "time_step": index,
            "previous_state": previous_pattern,
            "pattern": current_pattern_raw,
            "mapped_to": current_pattern,
            "status": "unseen" if is_unseen else "seen",
            "levenshtein_distance": max_distance,
            "transition": f"{previous_pattern}->{current_pattern}",
            "transition_probability": transition_probability,
            "path_probability": path_probability,
            "probability_threshold": probability_threshold,
            "anomaly_score": anomaly_score,
            "y_true": y_true,
            "y_pred": y_pred,
            "decision": "anomaly" if y_pred == 1 else "normal",
            "confidence_score": path_probability
        })

    return prediction_rows


def main():
    all_rows = []

    for fold in range(1, 6):
        all_rows.extend(
            generate_prediction_rows(
                train_pattern_path=f"data/processed/patterns/SKAB/fold_{fold}/train.csv",
                test_pattern_path=f"data/processed/patterns/SKAB/fold_{fold}/test.csv",
                label_path=f"data/processed/pca/SKAB/fold_{fold}/test.csv",
                label_column="anomaly",
                dataset="SKAB",
                fold=fold,
                scenario="normal"
            )
        )

    dataset_name = "BATADAL_dataset04"

    all_rows.extend(
        generate_prediction_rows(
            train_pattern_path=f"data/processed/patterns/BATADAL/{dataset_name}/train.csv",
            test_pattern_path=f"data/processed/patterns/BATADAL/{dataset_name}/test.csv",
            label_path=f"data/processed/pca/BATADAL/{dataset_name}/test.csv",
            label_column="ATT_FLAG",
            dataset=dataset_name,
            fold="-",
            scenario="normal"
        )
    )

    output_path = OUTPUT_DIR / "automata_prediction_outputs.csv"

    pd.DataFrame(all_rows).to_csv(output_path, index=False)

    print(f"Prediction-level automata outputs saved to: {output_path}")
    print(f"Total prediction rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
