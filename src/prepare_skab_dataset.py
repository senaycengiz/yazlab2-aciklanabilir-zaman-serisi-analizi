import os
import pandas as pd


RAW_SKAB_PATH = "data/raw/SKAB"
OUTPUT_DIR = "data/processed/SKAB"
OUTPUT_FILE = "skab_combined.csv"


def load_skab_group(group_name):
    group_path = os.path.join(RAW_SKAB_PATH, group_name)
    dataframes = []

    for file_name in sorted(os.listdir(group_path)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(group_path, file_name)

            df = pd.read_csv(file_path, sep=";")

            df["source_group"] = group_name
            df["source_file"] = file_name

            dataframes.append(df)

    return dataframes


def prepare_skab_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_dataframes = []

    for group_name in ["valve1", "valve2"]:
        group_dataframes = load_skab_group(group_name)
        all_dataframes.extend(group_dataframes)

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    combined_df.to_csv(output_path, index=False)

    print("SKAB veri seti başarıyla birleştirildi.")
    print(f"Çıktı dosyası: {output_path}")
    print(f"Toplam satır sayısı: {combined_df.shape[0]}")
    print(f"Toplam sütun sayısı: {combined_df.shape[1]}")
    print("Eklenen takip sütunları: source_group, source_file")

    print("\nsource_group dağılımı:")
    print(combined_df["source_group"].value_counts())

    print("\nİlk 5 satır:")
    print(combined_df.head())


if __name__ == "__main__":
    prepare_skab_dataset()
