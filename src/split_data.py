import os
import pandas as pd

from sklearn.model_selection import GroupKFold

from config.config import TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO
from src.data_loader import load_csv, load_skab_dataset


def split_time_series_data(df):
    """
    BATADAL gibi zaman serisi yapısı korunması gereken veri setlerini
    sıralı şekilde train, validation ve test olarak böler.
    Shuffle kullanılmaz.
    """

    total_ratio = TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO

    if round(total_ratio, 2) != 1.00:
        raise ValueError("Train, validation ve test oranlarının toplamı 1 olmalıdır.")

    total_rows = len(df)

    train_end = int(total_rows * TRAIN_RATIO)
    val_end = train_end + int(total_rows * VALIDATION_RATIO)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def save_batadal_split():
    """
    BATADAL için yalnızca Training Dataset 2 kullanılır.
    Veri zaman sırası korunarak %60 train, %20 validation, %20 test olarak bölünür.
    """

    dataset_name = "BATADAL"
    file_name = "BATADAL_dataset04.csv"
    file_path = f"data/raw/{dataset_name}/{file_name}"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} bulunamadı. BATADAL Training Dataset 2 dosyasını bu konuma eklemelisin.")

    df = load_csv(file_path)

    output_dir = "data/processed/BATADAL/BATADAL_dataset04"
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, test_df = split_time_series_data(df)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/validation.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print("\nBATADAL - Training Dataset 2 veri seti bölündü:")
    print(f"Train: {len(train_df)} satır")
    print(f"Validation: {len(val_df)} satır")
    print(f"Test: {len(test_df)} satır")
    print(f"Toplam: {len(train_df) + len(val_df) + len(test_df)} satır")


def save_skab_group_splits(n_splits=5):
    """
    SKAB için source_file grup değişkeni olarak kullanılır.
    Aynı CSV dosyasına ait kayıtların train ve testte birlikte yer alması engellenir.
    GroupKFold ile fold bazlı ayrım yapılır.
    """

    df = load_skab_dataset("data/raw/SKAB")

    if "source_file" not in df.columns:
        raise ValueError("SKAB veri setinde source_file sütunu bulunamadı.")

    groups = df["source_file"]

    unique_group_count = groups.nunique()
    if unique_group_count < n_splits:
        n_splits = unique_group_count

    group_kfold = GroupKFold(n_splits=n_splits)

    output_base_dir = "data/processed/SKAB"
    os.makedirs(output_base_dir, exist_ok=True)

    for fold_index, (train_index, test_index) in enumerate(group_kfold.split(df, groups=groups), start=1):
        fold_dir = f"{output_base_dir}/fold_{fold_index}"
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.iloc[train_index].copy()
        test_df = df.iloc[test_index].copy()

        train_df.to_csv(f"{fold_dir}/train.csv", index=False)
        test_df.to_csv(f"{fold_dir}/test.csv", index=False)

        train_files = set(train_df["source_file"].unique())
        test_files = set(test_df["source_file"].unique())
        common_files = train_files.intersection(test_files)

        if common_files:
            raise ValueError(f"Fold {fold_index} için veri sızıntısı var: {common_files}")

        print(f"\nSKAB fold_{fold_index} oluşturuldu:")
        print(f"Train: {len(train_df)} satır")
        print(f"Test: {len(test_df)} satır")
        print(f"Train dosya sayısı: {len(train_files)}")
        print(f"Test dosya sayısı: {len(test_files)}")


if __name__ == "__main__":
    save_batadal_split()
    save_skab_group_splits()