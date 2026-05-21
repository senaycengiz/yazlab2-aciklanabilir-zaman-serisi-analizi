import os
import pandas as pd


PATTERN_DATA_DIR = "data/processed/patterns"
TRANSITION_MATRIX_DIR = "results/transition_matrices"
TRANSITION_PROBABILITY_DIR = "results/transition_probabilities"


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


def create_transition_probability_matrix(transition_matrix, smoothing_alpha=1.0):
    """
    Transition count matrixini olasılık matrixine dönüştürür.

    Frekans tabanlı formül:
    P(to_state | from_state) = count(from_state -> to_state) / total_outgoing_count(from_state)

    Smoothing formülü:
    P(to_state | from_state) =
    (count(from_state -> to_state) + alpha) /
    (total_outgoing_count(from_state) + alpha * state_count)

    Bu sayede hiç görülmeyen geçişler 0 olmaz.
    Zero probability problemi önlenir.
    """
    state_count = transition_matrix.shape[1]

    smoothed_matrix = transition_matrix + smoothing_alpha

    row_sums = transition_matrix.sum(axis=1) + (smoothing_alpha * state_count)

    probability_matrix = smoothed_matrix.div(row_sums, axis=0)

    probability_matrix = probability_matrix.fillna(0)

    return probability_matrix

def analyze_transition_matrix(input_path, matrix_output_path, probability_output_path):
    """
    Tek bir pattern dosyası için transition count matrix ve
    transition probability matrix üretir.
    """
    df = pd.read_csv(input_path)

    if "pattern" not in df.columns:
        raise ValueError(f"{input_path} dosyasında pattern kolonu bulunamadı.")

    patterns = df["pattern"].astype(str).tolist()

    transition_matrix = create_transition_matrix(patterns)
    probability_matrix = create_transition_probability_matrix(transition_matrix)

    os.makedirs(os.path.dirname(matrix_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(probability_output_path), exist_ok=True)

    transition_matrix.to_csv(matrix_output_path)
    probability_matrix.to_csv(probability_output_path)

    print(f"Transition count matrix kaydedildi: {matrix_output_path}")
    print(f"Transition probability matrix kaydedildi: {probability_output_path}")
    print(f"State sayısı: {transition_matrix.shape[0]}")
    print(f"Matrix boyutu: {transition_matrix.shape}")


def process_all_pattern_files():
    """
    Tüm pattern dosyaları için transition count ve probability matrix üretir.
    """
    if not os.path.exists(PATTERN_DATA_DIR):
        print(f"Atlandı, pattern klasörü yok: {PATTERN_DATA_DIR}")
        return

    for root, dirs, files in os.walk(PATTERN_DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(input_path, PATTERN_DATA_DIR)

                matrix_output_path = os.path.join(
                    TRANSITION_MATRIX_DIR,
                    relative_path
                )

                probability_output_path = os.path.join(
                    TRANSITION_PROBABILITY_DIR,
                    relative_path
                )

                analyze_transition_matrix(
                    input_path=input_path,
                    matrix_output_path=matrix_output_path,
                    probability_output_path=probability_output_path
                )


if __name__ == "__main__":
    process_all_pattern_files()