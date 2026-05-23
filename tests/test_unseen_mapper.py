from src.unseen_mapper import map_unseen_pattern


def test_seen_pattern_maps_to_itself():
    known_patterns = ["abc", "bca", "ccc"]

    result = map_unseen_pattern("abc", known_patterns)

    assert result["original_pattern"] == "abc"
    assert result["is_unseen"] is False
    assert result["mapped_to"] == "abc"
    assert result["distance"] == 0


def test_unseen_pattern_maps_to_nearest_pattern():
    known_patterns = ["abc", "aaa", "ccc"]

    result = map_unseen_pattern("abd", known_patterns)

    assert result["original_pattern"] == "abd"
    assert result["is_unseen"] is True
    assert result["mapped_to"] == "abc"
    assert result["distance"] == 1