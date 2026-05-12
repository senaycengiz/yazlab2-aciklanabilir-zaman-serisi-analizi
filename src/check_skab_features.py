import pandas as pd


SKAB_COMBINED_PATH = "data/processed/SKAB/skab_combined.csv"

TARGET_COLUMN = "anomaly"

EXCLUDED_COLUMNS = [
    "datetime",
    "changepoint",
    "source_group",
    "source_file",
    TARGET_COLUMN
]


def check_skab_features():
    df = pd.read_csv(SKAB_COMBINED_PATH)

    feature_columns = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    print("SKAB feature kontrolü tamamlandı.")
    print(f"Toplam sütun sayısı: {len(df.columns)}")
    print(f"Model girdisi olarak kullanılacak feature sayısı: {len(feature_columns)}")
    print(f"Hedef değişken: {TARGET_COLUMN}")

    print("\nModel girdisine dahil EDİLMEYEN sütunlar:")
    for col in EXCLUDED_COLUMNS:
        if col in df.columns:
            print(f"- {col}")

    print("\nModel girdisi olarak kullanılacak sütunlar:")
    for col in feature_columns:
        print(f"- {col}")

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    check_skab_features()
