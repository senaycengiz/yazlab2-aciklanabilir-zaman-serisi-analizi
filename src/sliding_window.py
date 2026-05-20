import os
import pandas as pd

from config.config import WINDOW_SIZE, SAX_DATA_DIR, SAX_WORD_COLUMN


PATTERN_DATA_DIR = "data/processed/patterns"


def create_patterns(sax_values, window_size):
    """
    SAX sembollerinden sliding window yöntemi ile pattern listesi üretir.

    Örnek:
    sax_values = ["a", "b", "c", "d"]
    window_size = 3

    çıktı:
    ["abc", "bcd"]
    """
    if len(sax_values) < window_size:
        return []

    patterns = []

    for i in range(len(sax_values) - window_size + 1):
        pattern = "".join(sax_values[i:i + window_size])
        patterns.append(pattern)

    return patterns


def apply_sliding_window(input_path, output_path, window_size=WINDOW_SIZE):
    """
    Tek bir SAX CSV dosyasına sliding window uygular.
    Girdi dosyasında sax_word kolonu beklenir.
    Çıktı dosyasına pattern kolonu yazılır.
    """
    df = pd.read_csv(input_path)

    if SAX_WORD_COLUMN not in df.columns:
        raise ValueError(f"{input_path} dosyasında {SAX_WORD_COLUMN} kolonu bulunamadı.")

    sax_values = df[SAX_WORD_COLUMN].astype(str).tolist()
    patterns = create_patterns(sax_values, window_size)

    pattern_df = pd.DataFrame({
        "pattern": patterns
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pattern_df.to_csv(output_path, index=False)

    print(f"Sliding window tamamlandı: {output_path}")
    print(f"Window size: {window_size}")
    print(f"Pattern sayısı: {len(patterns)}")


def process_all_sax_files(window_size=WINDOW_SIZE):
    """
    data/processed/sax klasörü altındaki tüm SAX dosyalarını gezer.
    Her CSV dosyası için data/processed/patterns altında pattern dosyası üretir.
    """
    if not os.path.exists(SAX_DATA_DIR):
        print(f"Atlandı, SAX klasörü yok: {SAX_DATA_DIR}")
        return

    for root, dirs, files in os.walk(SAX_DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(input_path, SAX_DATA_DIR)
                output_path = os.path.join(PATTERN_DATA_DIR, relative_path)

                apply_sliding_window(
                    input_path=input_path,
                    output_path=output_path,
                    window_size=window_size
                )


if __name__ == "__main__":
    process_all_sax_files()