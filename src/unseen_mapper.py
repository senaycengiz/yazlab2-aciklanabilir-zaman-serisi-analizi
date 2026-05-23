from src.levenshtein import find_nearest_pattern


def map_unseen_pattern(pattern: str, known_patterns: list[str]) -> dict:
    """
    Pattern train sırasında görülmüşse direkt kendisine eşlenir.
    Görülmemişse Levenshtein ile en yakın pattern'e map edilir.
    """

    if pattern in known_patterns:
        return {
            "original_pattern": pattern,
            "is_unseen": False,
            "mapped_to": pattern,
            "distance": 0
        }

    nearest_pattern, distance = find_nearest_pattern(pattern, known_patterns)

    return {
        "original_pattern": pattern,
        "is_unseen": True,
        "mapped_to": nearest_pattern,
        "distance": distance
    }