import logging
import os
from datetime import datetime

from src.split_data import save_batadal_split, save_skab_group_splits
from src.normalization import run_normalization
from src.pca_transform import run_pca


os.makedirs("logs", exist_ok=True)

log_file = f"logs/preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)


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
    logging.info("Preprocessing pipeline başlatıldı.")

    try:

        print("\n1. Adım: Veri bölme işlemi başlatıldı.")
        logging.info("Veri bölme işlemi başlatıldı.")

        save_batadal_split()
        save_skab_group_splits()

        print("Veri bölme işlemi tamamlandı.")
        logging.info("Veri bölme işlemi tamamlandı.")

        print("\n2. Adım: Normalizasyon işlemi başlatıldı.")
        logging.info("Normalizasyon işlemi başlatıldı.")

        run_normalization()

        print("Normalizasyon işlemi tamamlandı.")
        logging.info("Normalizasyon işlemi tamamlandı.")

        print("\n3. Adım: PCA işlemi başlatıldı.")
        logging.info("PCA işlemi başlatıldı.")

        run_pca()

        print("PCA işlemi tamamlandı.")
        logging.info("PCA işlemi tamamlandı.")

        print("\nPreprocessing pipeline başarıyla tamamlandı.")
        logging.info("Preprocessing pipeline başarıyla tamamlandı.")

    except Exception as e:
        logging.error(f"Pipeline sırasında hata oluştu: {e}")
        raise


if __name__ == "__main__":
    run_preprocessing_pipeline()