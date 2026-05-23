import pandas as pd


def levenshtein_distance(a, b):
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

            current_row.append(min(insert_cost, delete_cost, replace_cost))

        previous_row = current_row

    return previous_row[-1]


def find_closest_pattern(sequence, known_patterns):
    if not known_patterns:
        raise ValueError("known_patterns listesi boş olamaz.")

    closest_pattern = None
    min_distance = float("inf")

    for pattern in known_patterns:
        distance = levenshtein_distance(sequence, pattern)

        if distance < min_distance:
            min_distance = distance
            closest_pattern = pattern

    return closest_pattern, min_distance


def map_unseen_pattern(sequence, known_patterns):
    """
    Sequence train'de varsa direkt kendisini kullanır.
    Train'de yoksa Levenshtein ile en yakın pattern'e eşler.
    """

    if not known_patterns:
        raise ValueError("known_patterns listesi boş olamaz.")

    if sequence in known_patterns:
        return {
            "original_pattern": sequence,
            "mapped_pattern": sequence,
            "is_unseen": False,
            "distance": 0
        }

    closest_pattern, distance = find_closest_pattern(sequence, known_patterns)

    return {
        "original_pattern": sequence,
        "mapped_pattern": closest_pattern,
        "is_unseen": True,
        "distance": distance
    }


def load_known_patterns_from_states(state_file_path):
    df = pd.read_csv(state_file_path)

    if "pattern" not in df.columns:
        raise ValueError(f"{state_file_path} dosyasında pattern kolonu bulunamadı.")

    return df["pattern"].astype(str).dropna().unique().tolist()


def predict_sequence(sequence, known_patterns, threshold=1, return_details=False):
    """
    Unseen pattern varsa önce en yakın pattern'e map eder.
    Sonra automata tahminini mapped pattern üzerinden yapar.
    """

    mapping_result = map_unseen_pattern(sequence, known_patterns)
    mapped_pattern = mapping_result["mapped_pattern"]
    distance = mapping_result["distance"]

    if mapped_pattern in known_patterns and distance <= threshold:
        prediction = 0
    else:
        prediction = 1

    if return_details:
        return {
            "sequence": sequence,
            "mapped_pattern": mapped_pattern,
            "is_unseen": mapping_result["is_unseen"],
            "distance": distance,
            "prediction": prediction
        }

    return prediction


def predict_from_state_file(sequence, state_file_path, threshold=1, return_details=False):
    known_patterns = load_known_patterns_from_states(state_file_path)

    return predict_sequence(
        sequence=sequence,
        known_patterns=known_patterns,
        threshold=threshold,
        return_details=return_details
    )