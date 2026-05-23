from src.levenshtein import levenshtein_distance, find_nearest_pattern


def test_levenshtein_same_patterns():
    assert levenshtein_distance("abc", "abc") == 0


def test_levenshtein_different_patterns():
    assert levenshtein_distance("abc", "abd") == 1


def test_find_nearest_pattern():
    known_patterns = ["aaa", "abc", "ccc"]
    nearest, distance = find_nearest_pattern("abd", known_patterns)

    assert nearest == "abc"
    assert distance == 1