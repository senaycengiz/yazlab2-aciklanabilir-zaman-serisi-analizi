from src.split_data import save_batadal_split, save_skab_group_splits
from src.normalization import run_normalization
from src.pca_transform import run_pca


def run_preprocessing_pipeline():
    """
    Tüm preprocessing adımlarını sırasıyla çalıştırır.

    Pipeline akışı:
    1. BATADAL zaman sıralı şekilde train / validation / test olarak bölünür.
    2. SKAB source_file bazlı GroupKFold ile foldlara ayrılır.
    3. Bölünen veriler StandardScaler ile normalize edilir.
    4. Normalize edilen verilere PCA uygulanır.
    """

    print("\nPreprocessing pipeline başlatıldı.")

    print("\n1. Adım: Veri bölme işlemi başlatıldı.")
    save_batadal_split()
    save_skab_group_splits()
    print("Veri bölme işlemi tamamlandı.")

    print("\n2. Adım: Normalizasyon işlemi başlatıldı.")
    run_normalization()
    print("Normalizasyon işlemi tamamlandı.")

    print("\n3. Adım: PCA işlemi başlatıldı.")
    run_pca()
    print("PCA işlemi tamamlandı.")

    print("\nPreprocessing pipeline başarıyla tamamlandı.")


if __name__ == "__main__":
    run_preprocessing_pipeline()