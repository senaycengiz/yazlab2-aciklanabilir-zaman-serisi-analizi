import os
import json
import numpy as np
import pandas as pd


ALPHABET_SIZE = 3
SAX_ALPHABET = ["a", "b", "c"]

PAA_DATA_DIR = "data/processed/paa"
SAX_DATA_DIR = "data/processed/sax"
SAX_WORD_COLUMN = "sax_word"


def fit_sax_breakpoints(train_df):
    train_values = train_df["PAA"].values

    quantiles = np.linspace(0, 1, ALPHABET_SIZE + 1)[1:-1]
    breakpoints = np.quantile(train_values, quantiles)

    return breakpoints


def value_to_symbol(value, breakpoints):
    symbol_index = np.digitize(value, breakpoints)
    return SAX_ALPHABET[symbol_index]


def transform_to_sax(df, breakpoints):
    sax_df = df.copy()

    sax_df[SAX_WORD_COLUMN] = sax_df["PAA"].apply(
        lambda value: value_to_symbol(value, breakpoints)
    )

    return sax_df


def create_sax_dictionary(train_sax_df):
    sax_words = train_sax_df[SAX_WORD_COLUMN].unique().tolist()

    sax_dictionary = {
        word: index
        for index, word in enumerate(sorted(sax_words))
    }

    return sax_dictionary


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def process_split_folder(input_dir, output_dir):
    train_path = os.path.join(input_dir, "train.csv")

    if not os.path.exists(train_path):
        print(f"Train dosyası bulunamadı, atlandı: {train_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)

    if "PAA" not in train_df.columns:
        print(f"PAA sütunu bulunamadı: {train_path}")
        return

    breakpoints = fit_sax_breakpoints(train_df)

    train_sax_df = None

    for split_name in ["train", "validation", "test"]:
        split_path = os.path.join(input_dir, f"{split_name}.csv")

        if not os.path.exists(split_path):
            continue

        df = pd.read_csv(split_path)

        if "PAA" not in df.columns:
            print(f"PAA sütunu bulunamadı: {split_path}")
            continue

        sax_df = transform_to_sax(df, breakpoints)

        output_path = os.path.join(output_dir, f"{split_name}.csv")
        sax_df.to_csv(output_path, index=False)

        print(f"SAX kaydedildi: {output_path}")

        if split_name == "train":
            train_sax_df = sax_df

    sax_dictionary = create_sax_dictionary(train_sax_df)

    save_json(
        sax_dictionary,
        os.path.join(output_dir, "sax_dictionary.json")
    )

    save_json(
        breakpoints.tolist(),
        os.path.join(output_dir, "sax_breakpoints.json")
    )

    print(f"SAX sözlüğü kaydedildi: {output_dir}")


def process_dataset(dataset_name):
    dataset_input_dir = os.path.join(PAA_DATA_DIR, dataset_name)
    dataset_output_dir = os.path.join(SAX_DATA_DIR, dataset_name)

    direct_train_path = os.path.join(dataset_input_dir, "train.csv")

    if os.path.exists(direct_train_path):
        process_split_folder(dataset_input_dir, dataset_output_dir)
        return

    subfolders = [
        name for name in os.listdir(dataset_input_dir)
        if os.path.isdir(os.path.join(dataset_input_dir, name))
    ]

    for subfolder in subfolders:
        input_dir = os.path.join(dataset_input_dir, subfolder)
        output_dir = os.path.join(dataset_output_dir, subfolder)

        print(f"{dataset_name} - {subfolder} işleniyor.")
        process_split_folder(input_dir, output_dir)


def main():
    print("SAX dönüşümü başlatıldı.")

    dataset_names = [
        name for name in os.listdir(PAA_DATA_DIR)
        if os.path.isdir(os.path.join(PAA_DATA_DIR, name))
    ]

    for dataset_name in dataset_names:
        print(f"\nİşleniyor: {dataset_name}")
        process_dataset(dataset_name)

    print("\nSAX dönüşümü tamamlandı.")


if __name__ == "__main__":
    main()