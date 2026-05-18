import os
import pandas as pd


def compare_models(metrics_files, output_path):
    """
    LSTM, GRU ve CNN modellerinin sonuçlarını tablo halinde karşılaştırır.
    """
    dfs = [pd.read_csv(file) for file in metrics_files]
    comparison_df = pd.concat(dfs, ignore_index=True)

    comparison_df = comparison_df.sort_values(
        by=["dataset", "f1_score"],
        ascending=[True, False]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison_df.to_csv(output_path, index=False)

    print(f"Model karşılaştırma tablosu kaydedildi: {output_path}")
    return comparison_df


def analyze_overfitting(train_metrics_path, test_metrics_path, output_path):
    """
    Train ve test metriklerini karşılaştırarak overfitting yorumu üretir.
    """
    train_df = pd.read_csv(train_metrics_path)
    test_df = pd.read_csv(test_metrics_path)

    rows = []

    for _, train_row in train_df.iterrows():
        dataset = train_row["dataset"]
        model = train_row["model"]

        test_row = test_df[
            (test_df["dataset"] == dataset) &
            (test_df["model"] == model)
        ]

        if test_row.empty:
            continue

        test_row = test_row.iloc[0]

        train_f1 = train_row["f1_score"]
        test_f1 = test_row["f1_score"]
        difference = train_f1 - test_f1

        if difference > 0.15:
            comment = "Overfitting olabilir. Model train verisini iyi öğrenmiş ama testte düşmüş."
        elif difference < -0.05:
            comment = "Test sonucu train sonucundan daha yüksek. Veri dağılımı farkı olabilir."
        else:
            comment = "Belirgin overfitting görünmüyor."

        rows.append({
            "dataset": dataset,
            "model": model,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "difference": difference,
            "comment": comment
        })

    result = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Overfitting analizi kaydedildi: {output_path}")
    return result