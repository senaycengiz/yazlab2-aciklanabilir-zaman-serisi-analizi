import os
import numpy as np
import pandas as pd
from config.config import RANDOM_SEED

NOISE_STD = 0.05



def add_gaussian_noise_to_numeric_columns(df, label_columns=None):
    if label_columns is None:
        label_columns = ["label", "anomaly", "ATT_FLAG", "target"]

    noisy_df = df.copy()

    numeric_columns = noisy_df.select_dtypes(include=[np.number]).columns

    feature_columns = [
        col for col in numeric_columns
        if col not in label_columns
    ]

    np.random.seed(RANDOM_SEED)

    for col in feature_columns:
        noise = np.random.normal(
            loc=0,
            scale=NOISE_STD,
            size=len(noisy_df)
        )
        noisy_df[col] = noisy_df[col] + noise

    return noisy_df


def create_noisy_file(input_path, output_path):
    df = pd.read_csv(input_path)

    noisy_df = add_gaussian_noise_to_numeric_columns(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    noisy_df.to_csv(output_path, index=False)

    print(f"Noisy veri oluşturuldu: {output_path}")


def main():
    input_files = [
        "data/processed/pca/BATADAL/BATADAL_dataset04/test.csv",
        "data/processed/pca/SKAB/fold_1/test.csv",
        "data/processed/pca/SKAB/fold_2/test.csv",
        "data/processed/pca/SKAB/fold_3/test.csv",
        "data/processed/pca/SKAB/fold_4/test.csv",
        "data/processed/pca/SKAB/fold_5/test.csv",
    ]

    for input_path in input_files:
        if not os.path.exists(input_path):
            print(f"Dosya bulunamadı, atlandı: {input_path}")
            continue

        output_path = input_path.replace(
            "data/processed/pca",
            "data/noisy/pca"
        )

        create_noisy_file(input_path, output_path)


if __name__ == "__main__":
    main()
