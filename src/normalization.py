import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


PROCESSED_DATA_DIR = "data/processed"
NORMALIZED_DATA_DIR = "data/processed/normalized"
SCALER_DIR = "models/scalers"


def get_feature_columns(dataframe):
    excluded_columns = [
        "DATETIME", "Date", "DATE", "Time", "TIME",
        "Timestamp", "datetime",
        "ATT_FLAG", "Attack", "Label", "LABEL", "label",
        "class", "Class", "Normal/Attack",
        "anomaly", "Anomaly",
        "changepoint",
        "source_group",
        "source_file"
    ]

    return [
        column for column in dataframe.columns
        if column not in excluded_columns
    ]


def prepare_numeric_features(train_data, other_dataframes, feature_columns):
    """
    Feature kolonlarını sayısala çevirir.
    Tamamen boş olan feature kolonlarını çıkarır.
    Eksik değerleri yalnızca train ortalaması ile doldurur.
    """

    for column in feature_columns:
        train_data[column] = pd.to_numeric(train_data[column], errors="coerce")

        for dataframe in other_dataframes:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    feature_columns = [
        column for column in feature_columns
        if not train_data[column].isna().all()
    ]

    if len(feature_columns) == 0:
        return []

    train_means = train_data[feature_columns].mean()

    train_data[feature_columns] = train_data[feature_columns].fillna(train_means)

    for dataframe in other_dataframes:
        dataframe[feature_columns] = dataframe[feature_columns].fillna(train_means)

    return feature_columns


def normalize_train_validation_test(train_path, validation_path, test_path, dataset_group, dataset_name):
    os.makedirs(NORMALIZED_DATA_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)

    train_data = pd.read_csv(train_path, low_memory=False)
    validation_data = pd.read_csv(validation_path, low_memory=False)
    test_data = pd.read_csv(test_path, low_memory=False)

    feature_columns = get_feature_columns(train_data)

    feature_columns = prepare_numeric_features(
        train_data=train_data,
        other_dataframes=[validation_data, test_data],
        feature_columns=feature_columns
    )

    if len(feature_columns) == 0:
        print(f"{dataset_group}/{dataset_name} için sayısal feature bulunamadı, atlandı.")
        return

    scaler = StandardScaler()

    train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    validation_data[feature_columns] = scaler.transform(validation_data[feature_columns])
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])

    dataset_output_dir = os.path.join(NORMALIZED_DATA_DIR, dataset_group, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    train_data.to_csv(os.path.join(dataset_output_dir, "train.csv"), index=False)
    validation_data.to_csv(os.path.join(dataset_output_dir, "validation.csv"), index=False)
    test_data.to_csv(os.path.join(dataset_output_dir, "test.csv"), index=False)

    scaler_output_path = os.path.join(SCALER_DIR, f"{dataset_group}_{dataset_name}_scaler.pkl")
    joblib.dump(scaler, scaler_output_path)

    print(f"{dataset_group}/{dataset_name} için normalizasyon tamamlandı.")
    print(f"Scaler: {scaler_output_path}")


def normalize_train_test(train_path, test_path, dataset_group, dataset_name):
    """
    SKAB fold yapısı için train/test normalizasyonu yapar.
    Scaler yalnızca train verisine fit edilir.
    """

    os.makedirs(NORMALIZED_DATA_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)

    train_data = pd.read_csv(train_path, low_memory=False)
    test_data = pd.read_csv(test_path, low_memory=False)

    feature_columns = get_feature_columns(train_data)

    feature_columns = prepare_numeric_features(
        train_data=train_data,
        other_dataframes=[test_data],
        feature_columns=feature_columns
    )

    if len(feature_columns) == 0:
        print(f"{dataset_group}/{dataset_name} için sayısal feature bulunamadı, atlandı.")
        return

    scaler = StandardScaler()

    train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])

    dataset_output_dir = os.path.join(NORMALIZED_DATA_DIR, dataset_group, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    train_data.to_csv(os.path.join(dataset_output_dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(dataset_output_dir, "test.csv"), index=False)

    scaler_output_path = os.path.join(SCALER_DIR, f"{dataset_group}_{dataset_name}_scaler.pkl")
    joblib.dump(scaler, scaler_output_path)

    print(f"{dataset_group}/{dataset_name} için normalizasyon tamamlandı.")
    print(f"Scaler: {scaler_output_path}")


def run_normalization():
    for dataset_group in os.listdir(PROCESSED_DATA_DIR):
        dataset_group_path = os.path.join(PROCESSED_DATA_DIR, dataset_group)

        if not os.path.isdir(dataset_group_path):
            continue

        if dataset_group in ["normalized", "pca"]:
            continue

        for dataset_name in os.listdir(dataset_group_path):
            dataset_path = os.path.join(dataset_group_path, dataset_name)

            if not os.path.isdir(dataset_path):
                continue

            train_path = os.path.join(dataset_path, "train.csv")
            validation_path = os.path.join(dataset_path, "validation.csv")
            test_path = os.path.join(dataset_path, "test.csv")

            if os.path.exists(train_path) and os.path.exists(validation_path) and os.path.exists(test_path):
                normalize_train_validation_test(
                    train_path=train_path,
                    validation_path=validation_path,
                    test_path=test_path,
                    dataset_group=dataset_group,
                    dataset_name=dataset_name
                )

            elif os.path.exists(train_path) and os.path.exists(test_path):
                normalize_train_test(
                    train_path=train_path,
                    test_path=test_path,
                    dataset_group=dataset_group,
                    dataset_name=dataset_name
                )

            else:
                print(f"{dataset_group}/{dataset_name} için gerekli split dosyaları eksik, atlandı.")


if __name__ == "__main__":
    run_normalization()