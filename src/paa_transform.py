import os
import numpy as np
import pandas as pd


PAA_SEGMENT_SIZE = 10


def apply_paa(series, segment_size):
    """
    Zaman serisini segmentlere böler
    ve her segmentin ortalamasını alır.
    """

    paa_values = []

    for i in range(0, len(series), segment_size):

        segment = series[i:i + segment_size]

        if len(segment) == segment_size:
            paa_values.append(np.mean(segment))

    return paa_values


def process_file(input_path, output_path):

    df = pd.read_csv(input_path)

    if "PC1" not in df.columns:
        print(f"PC1 sütunu bulunamadı: {input_path}")
        return

    pc1_series = df["PC1"].values

    paa_result = apply_paa(pc1_series, PAA_SEGMENT_SIZE)

    paa_df = pd.DataFrame({
        "PAA": paa_result
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    paa_df.to_csv(output_path, index=False)

    print(f"PAA kaydedildi: {output_path}")


def process_directory(input_dir, output_dir):

    for root, dirs, files in os.walk(input_dir):

        for file in files:

            if file.endswith(".csv") and file in ["train.csv", "validation.csv", "test.csv"]:

                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(input_path, input_dir)

                output_path = os.path.join(output_dir, relative_path)

                process_file(input_path, output_path)


if __name__ == "__main__":

    INPUT_DIR = "data/processed/pca"

    OUTPUT_DIR = "data/processed/paa"

    process_directory(INPUT_DIR, OUTPUT_DIR)

    print("PAA dönüşümü tamamlandı.")