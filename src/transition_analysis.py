import os
import pandas as pd


PATTERN_DATA_DIR = "data/processed/patterns"
TRANSITION_MATRIX_DIR = "results/transition_matrices"


def calculate_transition_counts(patterns):
    """
    Pattern dizisinde hangi patterndan hangi patterne kaç defa geçildiğini hesaplar.
    """
    transitions = {}

    for i in range(len(patterns) - 1):
        from_pattern = patterns[i]
        to_pattern = patterns[i + 1]

        key = (from_pattern, to_pattern)
        transitions[key] = transitions.get(key, 0) + 1

    return transitions


def create_transition_matrix(patterns):
    """
    Transition sayılarını matrix formatına dönüştürür.

    Satır: başlangıç pattern/state
    Sütun: gidilen pattern/state
    Hücre: geçiş sayısı
    """
    transition_counts = calculate_transition_counts(patterns)
    unique_patterns = sorted(set(patterns))

    matrix = pd.DataFrame(
        0,
        index=unique_patterns,
        columns=unique_patterns
    )

    for (from_pattern, to_pattern), count in transition_counts.items():
        matrix.loc[from_pattern, to_pattern] = count

    return matrix


def analyze_transition_matrix(input_path, output_path):
    """
    Tek bir pattern dosyası için transition matrix üretir.
    """
    df = pd.read_csv(input_path)

    if "pattern" not in df.columns:
        raise ValueError(f"{input_path} dosyasında pattern kolonu bulunamadı.")

    patterns = df["pattern"].astype(str).tolist()
    matrix = create_transition_matrix(patterns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    matrix.to_csv(output_path)

    print(f"Transition matrix kaydedildi: {output_path}")
    print(f"State sayısı: {matrix.shape[0]}")
    print(f"Matrix boyutu: {matrix.shape}")


def process_all_pattern_files():
    """
    Tüm pattern dosyaları için transition matrix üretir.
    """
    if not os.path.exists(PATTERN_DATA_DIR):
        print(f"Atlandı, pattern klasörü yok: {PATTERN_DATA_DIR}")
        return

    for root, dirs, files in os.walk(PATTERN_DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(input_path, PATTERN_DATA_DIR)
                output_path = os.path.join(TRANSITION_MATRIX_DIR, relative_path)

                analyze_transition_matrix(
                    input_path=input_path,
                    output_path=output_path
                )


if __name__ == "__main__":
    process_all_pattern_files()