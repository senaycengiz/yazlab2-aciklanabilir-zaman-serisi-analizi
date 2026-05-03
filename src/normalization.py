import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


PROCESSED_DATA_DIR = "data/processed"
NORMALIZED_DATA_DIR = "data/processed/normalized"
SCALER_DIR = "models/scalers"


def get_feature_columns(dataframe):
    excluded_columns = ["DATETIME", "ATT_FLAG", "Attack", "Label", "label", "class"]

    return [
        column for column in dataframe.columns
        if column not in excluded_columns
    ]


def normalize_dataset(train_path, validation_path, test_path, dataset_group, dataset_name):
    os.makedirs(NORMALIZED_DATA_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)

    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(validation_path)
    test_data = pd.read_csv(test_path)

    feature_columns = get_feature_columns(train_data)

    scaler = StandardScaler()

    train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    validation_data[feature_columns] = scaler.transform(validation_data[feature_columns])
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])

    dataset_output_dir = os.path.join(NORMALIZED_DATA_DIR, dataset_group, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    train_output_path = os.path.join(dataset_output_dir, "train.csv")
    validation_output_path = os.path.join(dataset_output_dir, "validation.csv")
    test_output_path = os.path.join(dataset_output_dir, "test.csv")

    scaler_output_path = os.path.join(SCALER_DIR, f"{dataset_group}_{dataset_name}_scaler.pkl")

    train_data.to_csv(train_output_path, index=False)
    validation_data.to_csv(validation_output_path, index=False)
    test_data.to_csv(test_output_path, index=False)

    joblib.dump(scaler, scaler_output_path)

    print(f"{dataset_group}/{dataset_name} için normalizasyon tamamlandı.")
    print(f"Train: {train_output_path}")
    print(f"Validation: {validation_output_path}")
    print(f"Test: {test_output_path}")
    print(f"Scaler: {scaler_output_path}")


def run_normalization():
    for dataset_group in os.listdir(PROCESSED_DATA_DIR):
        dataset_group_path = os.path.join(PROCESSED_DATA_DIR, dataset_group)

        if not os.path.isdir(dataset_group_path):
            continue

        if dataset_group == "normalized":
            continue

        for dataset_name in os.listdir(dataset_group_path):
            dataset_path = os.path.join(dataset_group_path, dataset_name)

            if not os.path.isdir(dataset_path):
                continue

            train_path = os.path.join(dataset_path, "train.csv")
            validation_path = os.path.join(dataset_path, "validation.csv")
            test_path = os.path.join(dataset_path, "test.csv")

            if not (
                os.path.exists(train_path)
                and os.path.exists(validation_path)
                and os.path.exists(test_path)
            ):
                print(f"{dataset_group}/{dataset_name} için train/validation/test dosyaları eksik, atlandı.")
                continue

            normalize_dataset(
                train_path=train_path,
                validation_path=validation_path,
                test_path=test_path,
                dataset_group=dataset_group,
                dataset_name=dataset_name
            )


if __name__ == "__main__":
    run_normalization()