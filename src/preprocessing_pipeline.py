from src.split_data import process_dataset_folder
from src.normalization import run_normalization
from src.pca_transform import apply_pca, DATASETS


def run_preprocessing_pipeline():
    """
    Tüm preprocessing adımlarını sırasıyla çalıştırır.

    Pipeline akışı:
    1. Raw veriler train / validation / test olarak bölünür.
    2. Bölünen veriler StandardScaler ile normalize edilir.
    3. Normalize edilen verilere PCA uygulanır.
    """

    print("\nPreprocessing pipeline başlatıldı.")

    print("\n1. Adım: Veri bölme işlemi başlatıldı.")
    process_dataset_folder("BATADAL")
    process_dataset_folder("WADI")
    print("Veri bölme işlemi tamamlandı.")

    print("\n2. Adım: Normalizasyon işlemi başlatıldı.")
    run_normalization()
    print("Normalizasyon işlemi tamamlandı.")

    print("\n3. Adım: PCA işlemi başlatıldı.")
    for dataset_name, dataset_file in DATASETS:
        apply_pca(dataset_name, dataset_file)
    print("PCA işlemi tamamlandı.")

    print("\nPreprocessing pipeline başarıyla tamamlandı.")


if __name__ == "__main__":
    run_preprocessing_pipeline()
