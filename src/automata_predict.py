import pandas as pd


def levenshtein_distance(a, b):
    """
    İki string arasındaki Levenshtein mesafesini hesaplar.
    Mesafe küçükse patternler birbirine benzer demektir.
    """

    if len(a) < len(b):
        return levenshtein_distance(b, a)

    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current_row = [i]

        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1]

            if char_a != char_b:
                replace_cost += 1

            current_row.append(
                min(insert_cost, delete_cost, replace_cost)
            )

        previous_row = current_row

    return previous_row[-1]


def find_closest_pattern(sequence, known_patterns):
    """
    Gelen sequence'a en yakın eğitim patternini bulur.
    """

    closest_pattern = None
    min_distance = float("inf")

    for pattern in known_patterns:
        distance = levenshtein_distance(sequence, pattern)

        if distance < min_distance:
            min_distance = distance
            closest_pattern = pattern

    return closest_pattern, min_distance


def load_known_patterns_from_states(state_file_path):
    """
    Automata state dosyasından bilinen patternleri okur.
    """

    df = pd.read_csv(state_file_path)

    if "pattern" not in df.columns:
        raise ValueError(f"{state_file_path} dosyasında pattern kolonu bulunamadı.")

    patterns = df["pattern"].astype(str).tolist()

    return patterns


def predict_sequence(sequence, known_patterns, threshold=1):
    """
    Verilen sequence için otomata tabanlı tahmin yapar.

    Dönüş:
    0 -> normal
    1 -> anomaly
    """

    if sequence in known_patterns:
        return 0

    closest_pattern, distance = find_closest_pattern(
        sequence,
        known_patterns
    )

    if distance <= threshold:
        return 0

    return 1


def predict_from_state_file(sequence, state_file_path, threshold=1):
    """
    State CSV dosyasını kullanarak tek bir sequence için tahmin yapar.
    """

    known_patterns = load_known_patterns_from_states(state_file_path)

    prediction = predict_sequence(
        sequence=sequence,
        known_patterns=known_patterns,
        threshold=threshold
    )

    return prediction
