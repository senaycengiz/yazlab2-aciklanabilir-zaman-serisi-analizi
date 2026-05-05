import os
import pickle
import pandas as pd
from sklearn.decomposition import PCA

from config.config import PCA_COMPONENTS

DATASETS = [
    ("BATADAL", "BATADAL_dataset03"),
    ("BATADAL", "BATADAL_dataset04"),
    ("WADI", "WADI_14days_new"),
    ("WADI", "WADI_attackdataLABLE"),
]


def apply_pca(dataset_name, dataset_file):
    input_dir = f"data/processed/normalized/{dataset_name}/{dataset_file}"
    output_dir = f"data/processed/pca/{dataset_name}/{dataset_file}"
    model_dir = "models/pca"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    validation_df = pd.read_csv(os.path.join(input_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))

    non_feature_columns = [
        "DATETIME", "Date", "DATE", "Time", "TIME",
        "ATT_FLAG", "Attack", "Label", "LABEL", "label",
        "class", "Class", "Normal/Attack", "attack"
    ]

    # sadece sayısal kolonları al
    feature_columns = train_df.select_dtypes(include=["number"]).columns.tolist()

    feature_columns = [
        col for col in feature_columns
        if col not in non_feature_columns
    ]

    train_features = train_df[feature_columns]
    validation_features = validation_df[feature_columns]
    test_features = test_df[feature_columns]

    # 🔥 tamamen NaN olan kolonları sil
    train_features = train_features.dropna(axis=1, how="all")
    validation_features = validation_features[train_features.columns]
    test_features = test_features[train_features.columns]

    # 🔥 train ortalaması
    train_means = train_features.mean()

    # NaN doldur
    train_features = train_features.fillna(train_means)
    validation_features = validation_features.fillna(train_means)
    test_features = test_features.fillna(train_means)

    # son güvenlik (hala NaN varsa)
    train_features = train_features.fillna(0)
    validation_features = validation_features.fillna(0)
    test_features = test_features.fillna(0)

    pca = PCA(n_components=PCA_COMPONENTS)

    train_pc1 = pca.fit_transform(train_features)
    validation_pc1 = pca.transform(validation_features)
    test_pc1 = pca.transform(test_features)

    train_output = pd.DataFrame({"PC1": train_pc1[:, 0]})
    validation_output = pd.DataFrame({"PC1": validation_pc1[:, 0]})
    test_output = pd.DataFrame({"PC1": test_pc1[:, 0]})

    # label / datetime geri ekle
    for col in non_feature_columns:
        if col in train_df.columns:
            train_output[col] = train_df[col]
        if col in validation_df.columns:
            validation_output[col] = validation_df[col]
        if col in test_df.columns:
            test_output[col] = test_df[col]

    train_output.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    validation_output.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_output.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    pca_path = os.path.join(model_dir, f"{dataset_name}_{dataset_file}_pca.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)

    print(f"{dataset_name}/{dataset_file} için PCA tamamlandı.")
    print(f"Train: {train_output.shape}")
    print(f"Validation: {validation_output.shape}")
    print(f"Test: {test_output.shape}")
    print(f"PCA Model: {pca_path}")


if __name__ == "__main__":
    for dataset_name, dataset_file in DATASETS:
        apply_pca(dataset_name, dataset_file)