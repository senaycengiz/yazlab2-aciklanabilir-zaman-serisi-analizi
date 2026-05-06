import os
import pandas as pd


def load_csv(path):
    """
    Verilen CSV dosyasını pandas ile okur.
    SKAB dosyalarında ayraç ; olduğu için otomatik separator algılanır.
    Kolon isimlerindeki gereksiz boşluklar temizlenir.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df


def load_skab_dataset(base_path="data/raw/SKAB"):
    """
    SKAB veri setinde yalnızca valve1 ve valve2 klasörleri kullanılır.
    Bu klasörlerdeki tüm CSV dosyaları birleştirilir.

    Eklenen sütunlar:
    - source_group: valve1 veya valve2
    - source_file: kaydın geldiği CSV dosyası
    """
    all_dataframes = []

    for group_name in ["valve1", "valve2"]:
        group_path = os.path.join(base_path, group_name)

        if not os.path.exists(group_path):
            print(f"Uyarı: {group_path} klasörü bulunamadı.")
            continue

        for file_name in os.listdir(group_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(group_path, file_name)

                df = load_csv(file_path)
                df["source_group"] = group_name
                df["source_file"] = file_name

                all_dataframes.append(df)

    if not all_dataframes:
        raise FileNotFoundError("SKAB için valve1/valve2 klasörlerinde CSV dosyası bulunamadı.")

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df


def print_basic_info(df, dataset_name):
    """
    Veri setinin temel bilgilerini ekrana yazdırır.
    """
    print(f"\n===== {dataset_name} =====")
    print("Satır sayısı:", df.shape[0])
    print("Kolon sayısı:", df.shape[1])

    print("\nKolonlar:")
    print(df.columns)

    print("\nEksik veri sayıları:")
    print(df.isnull().sum())


def find_label_column(df):
    """
    Veri setindeki olası label/anomaly sütununu bulur.
    """
    possible_label_columns = [
        "ATT_FLAG",
        "Label",
        "label",
        "Attack",
        "attack",
        "Anomaly",
        "anomaly",
        "class",
        "target",
        "Attack LABLE (1:No Attack, -1:Attack)"
    ]

    for col in possible_label_columns:
        if col in df.columns:
            return col

    return None


def get_feature_columns(df):
    """
    Model girdisi olarak kullanılacak feature sütunlarını döndürür.
    Zaman, etiket ve takip amaçlı sütunlar çıkarılır.
    """
    label_col = find_label_column(df)

    excluded_columns = [
        "datetime",
        "DATETIME",
        "Date",
        "Time",
        "Timestamp",
        "changepoint",
        "source_group",
        "source_file"
    ]

    if label_col is not None:
        excluded_columns.append(label_col)

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    return feature_columns


def print_anomaly_info(df, dataset_name):
    """
    Label/anomaly sütunu varsa dağılımını ve yüzdesini yazdırır.
    """
    label_col = find_label_column(df)

    if label_col is None:
        print(f"\n{dataset_name} için label/anomaly sütunu bulunamadı.")
        return

    print(f"\nLabel/anomaly sütunu: {label_col}")

    print("\nAnomaly dağılımı:")
    print(df[label_col].value_counts())

    print("\nYüzdelik dağılım:")
    print(df[label_col].value_counts(normalize=True) * 100)