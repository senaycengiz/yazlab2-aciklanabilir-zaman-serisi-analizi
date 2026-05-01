import os
import pandas as pd

from config.config import TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO


def split_time_series_data(df):
    """
    Zaman serisi verisini sıralı şekilde train, validation ve test olarak böler.
    Shuffle kullanılmaz.
    """

    total_ratio = TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO

    if round(total_ratio, 2) != 1.00:
        raise ValueError("Train, validation ve test oranlarının toplamı 1 olmalıdır.")

    total_rows = len(df)

    train_end = int(total_rows * TRAIN_RATIO)
    val_end = train_end + int(total_rows * VALIDATION_RATIO)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def save_splits(dataset_name, source_file_name, df):
    """
    Bölünen verileri data/processed klasörüne kaydeder.
    Her csv dosyası için ayrı klasör oluşturur.
    """

    file_base_name = os.path.splitext(source_file_name)[0]
    output_dir = f"data/processed/{dataset_name}/{file_base_name}"
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, test_df = split_time_series_data(df)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/validation.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"\n{dataset_name} - {source_file_name} veri seti bölündü:")
    print(f"Train: {len(train_df)} satır")
    print(f"Validation: {len(val_df)} satır")
    print(f"Test: {len(test_df)} satır")
    print(f"Toplam: {len(train_df) + len(val_df) + len(test_df)} satır")


def process_dataset_folder(dataset_name):
    """
    data/raw içindeki veri seti klasörünü okur.
    Klasör içindeki tüm csv dosyalarını sıralı şekilde böler.
    """

    folder_path = f"data/raw/{dataset_name}"

    if not os.path.exists(folder_path):
        print(f"{dataset_name} klasörü bulunamadı.")
        return

    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

    if not csv_files:
        print(f"{dataset_name} klasöründe csv dosyası bulunamadı.")
        return

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)

        print(f"\nİşleniyor: {file_path}")

        df = pd.read_csv(file_path, low_memory=False)
        save_splits(dataset_name, csv_file, df)


if __name__ == "__main__":
    process_dataset_folder("BATADAL")
    process_dataset_folder("WADI")