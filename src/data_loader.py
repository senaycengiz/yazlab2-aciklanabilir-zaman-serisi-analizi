import pandas as pd


def load_csv(path):
    """
    Verilen CSV dosyasını pandas ile okur.
    Kolon isimlerindeki gereksiz boşlukları temizler.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def load_wadi_attack(path):
    """
    WADI attack dosyasında ilk satır gerçek kolon isimlerini içerdiği için
    dosya özel olarak düzenlenerek okunur.
    """
    df = pd.read_csv(path, low_memory=False)

    # İlk satırı kolon ismi olarak al
    new_columns = df.iloc[0].astype(str).str.strip()
    df = df.iloc[1:].copy()
    df.columns = new_columns

    return df


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