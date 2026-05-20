import os
import pandas as pd

from config.config import WINDOW_SIZE_OPTIONS, SAX_DATA_DIR, SAX_WORD_COLUMN
from src.sliding_window import create_patterns


def analyze_pattern_counts_for_values(sax_values):
    results = []

    for window_size in WINDOW_SIZE_OPTIONS:
        patterns = create_patterns(sax_values, window_size)

        results.append({
            "window_size": window_size,
            "sax_length": len(sax_values),
            "pattern_count": len(patterns),
            "unique_pattern_count": len(set(patterns))
        })

    return results


def analyze_pattern_counts(input_path):
    df = pd.read_csv(input_path)

    if SAX_WORD_COLUMN not in df.columns:
        raise ValueError(f"{input_path} dosyasında {SAX_WORD_COLUMN} kolonu bulunamadı.")

    sax_values = df[SAX_WORD_COLUMN].astype(str).tolist()
    results = analyze_pattern_counts_for_values(sax_values)

    for row in results:
        row["file"] = input_path

    return results


def analyze_all_sax_files():
    all_results = []

    if not os.path.exists(SAX_DATA_DIR):
        print(f"Atlandı, SAX klasörü yok: {SAX_DATA_DIR}")
        return

    for root, dirs, files in os.walk(SAX_DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                input_path = os.path.join(root, file)
                results = analyze_pattern_counts(input_path)
                all_results.extend(results)

    output_path = "results/pattern_count_analysis.csv"
    os.makedirs("results", exist_ok=True)

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(output_path, index=False)

    print(f"Pattern sayı analizi kaydedildi: {output_path}")
    print(result_df.head())


if __name__ == "__main__":
    analyze_all_sax_files()