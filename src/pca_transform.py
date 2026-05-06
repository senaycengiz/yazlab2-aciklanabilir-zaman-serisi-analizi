import os
import pickle
import pandas as pd
from sklearn.decomposition import PCA

from config.config import PCA_COMPONENTS


NORMALIZED_DATA_DIR = "data/processed/normalized"
PCA_OUTPUT_DIR = "data/processed/pca"
PCA_MODEL_DIR = "models/pca"


NON_FEATURE_COLUMNS = [
    "DATETIME", "Date", "DATE", "Time", "TIME",
    "Timestamp", "datetime",
    "ATT_FLAG", "Attack", "Label", "LABEL", "label",
    "class", "Class", "Normal/Attack", "attack",
    "anomaly", "Anomaly",
    "changepoint",
    "source_group",
    "source_file"
]


def get_numeric_feature_columns(train_df):
    feature_columns = train_df.select_dtypes(include=["number"]).columns.tolist()

    feature_columns = [
        col for col in feature_columns
        if col not in NON_FEATURE_COLUMNS
    ]

    return feature_columns


def prepare_pca_features(train_df, other_dataframes, feature_columns):
    train_features = train_df[feature_columns].copy()

    other_features = [
        dataframe[feature_columns].copy()
        for dataframe in other_dataframes
    ]

    train_features = train_features.dropna(axis=1, how="all")

    other_features = [
        features[train_features.columns]
        for features in other_features
    ]

    train_means = train_features.mean()

    train_features = train_features.fillna(train_means).fillna(0)

    other_features = [
        features.fillna(train_means).fillna(0)
        for features in other_features
    ]

    return train_features, other_features


def add_metadata_columns(output_df, original_df):
    for col in NON_FEATURE_COLUMNS:
        if col in original_df.columns:
            output_df[col] = original_df[col].values

    return output_df


def apply_pca_to_dataset(dataset_group, dataset_name):
    input_dir = os.path.join(NORMALIZED_DATA_DIR, dataset_group, dataset_name)
    output_dir = os.path.join(PCA_OUTPUT_DIR, dataset_group, dataset_name)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(PCA_MODEL_DIR, exist_ok=True)

    train_path = os.path.join(input_dir, "train.csv")
    validation_path = os.path.join(input_dir, "validation.csv")
    test_path = os.path.join(input_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"{dataset_group}/{dataset_name} için train/test dosyası eksik, PCA atlandı.")
        return

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    has_validation = os.path.exists(validation_path)

    if has_validation:
        validation_df = pd.read_csv(validation_path, low_memory=False)
        other_dataframes = [validation_df, test_df]
    else:
        validation_df = None
        other_dataframes = [test_df]

    feature_columns = get_numeric_feature_columns(train_df)

    if len(feature_columns) == 0:
        print(f"{dataset_group}/{dataset_name} için PCA uygulanacak sayısal feature bulunamadı.")
        return

    train_features, other_features = prepare_pca_features(
        train_df=train_df,
        other_dataframes=other_dataframes,
        feature_columns=feature_columns
    )

    if train_features.shape[1] == 0:
        print(f"{dataset_group}/{dataset_name} için kullanılabilir feature kalmadı, PCA atlandı.")
        return

    pca = PCA(n_components=PCA_COMPONENTS)

    train_pc = pca.fit_transform(train_features)

    train_output = pd.DataFrame({"PC1": train_pc[:, 0]})
    train_output = add_metadata_columns(train_output, train_df)

    train_output.to_csv(os.path.join(output_dir, "train.csv"), index=False)

    if has_validation:
        validation_features = other_features[0]
        test_features = other_features[1]

        validation_pc = pca.transform(validation_features)
        test_pc = pca.transform(test_features)

        validation_output = pd.DataFrame({"PC1": validation_pc[:, 0]})
        test_output = pd.DataFrame({"PC1": test_pc[:, 0]})

        validation_output = add_metadata_columns(validation_output, validation_df)
        test_output = add_metadata_columns(test_output, test_df)

        validation_output.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
        test_output.to_csv(os.path.join(output_dir, "test.csv"), index=False)

        print(f"{dataset_group}/{dataset_name} için PCA tamamlandı.")
        print(f"Train: {train_output.shape}")
        print(f"Validation: {validation_output.shape}")
        print(f"Test: {test_output.shape}")

    else:
        test_features = other_features[0]
        test_pc = pca.transform(test_features)

        test_output = pd.DataFrame({"PC1": test_pc[:, 0]})
        test_output = add_metadata_columns(test_output, test_df)

        test_output.to_csv(os.path.join(output_dir, "test.csv"), index=False)

        print(f"{dataset_group}/{dataset_name} için PCA tamamlandı.")
        print(f"Train: {train_output.shape}")
        print(f"Test: {test_output.shape}")

    pca_path = os.path.join(PCA_MODEL_DIR, f"{dataset_group}_{dataset_name}_pca.pkl")

    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)

    print(f"PCA Model: {pca_path}")


def run_pca():
    if not os.path.exists(NORMALIZED_DATA_DIR):
        print("Normalized veri klasörü bulunamadı, PCA çalıştırılamadı.")
        return

    for dataset_group in os.listdir(NORMALIZED_DATA_DIR):
        dataset_group_path = os.path.join(NORMALIZED_DATA_DIR, dataset_group)

        if not os.path.isdir(dataset_group_path):
            continue

        for dataset_name in os.listdir(dataset_group_path):
            dataset_path = os.path.join(dataset_group_path, dataset_name)

            if not os.path.isdir(dataset_path):
                continue

            apply_pca_to_dataset(dataset_group, dataset_name)


if __name__ == "__main__":
    run_pca()