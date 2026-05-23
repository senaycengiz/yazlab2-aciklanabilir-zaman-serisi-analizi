import pandas as pd

from src.automata_predict import (
    map_unseen_pattern,
    predict_sequence,
    predict_from_state_file
)


def test_seen_pattern_maps_to_itself():
    known_patterns = ["abc", "bca", "ccc"]

    result = map_unseen_pattern("abc", known_patterns)

    assert result["original_pattern"] == "abc"
    assert result["mapped_pattern"] == "abc"
    assert result["is_unseen"] is False
    assert result["distance"] == 0


def test_unseen_pattern_maps_to_closest_pattern():
    known_patterns = ["abc", "bca", "ccc"]

    result = map_unseen_pattern("abd", known_patterns)

    assert result["original_pattern"] == "abd"
    assert result["mapped_pattern"] == "abc"
    assert result["is_unseen"] is True
    assert result["distance"] == 1


def test_predict_sequence_with_unseen_pattern_returns_normal_if_close():
    known_patterns = ["abc", "bca", "ccc"]

    result = predict_sequence(
        sequence="abd",
        known_patterns=known_patterns,
        threshold=1,
        return_details=True
    )

    assert isinstance(result, dict)
    assert result["is_unseen"] is True
    assert result["mapped_pattern"] == "abc"
    assert result["distance"] == 1
    assert result["prediction"] == 0


def test_predict_sequence_with_far_unseen_pattern_returns_anomaly():
    known_patterns = ["abc", "bca", "ccc"]

    result = predict_sequence(
        sequence="zzz",
        known_patterns=known_patterns,
        threshold=1,
        return_details=True
    )

    assert isinstance(result, dict)
    assert result["is_unseen"] is True
    assert result["prediction"] == 1


def test_predict_from_state_file_with_unseen_pattern(tmp_path):
    state_file = tmp_path / "states.csv"

    df = pd.DataFrame({
        "state": [0, 1, 2],
        "pattern": ["abc", "bca", "ccc"]
    })

    df.to_csv(state_file, index=False)

    result = predict_from_state_file(
        sequence="abd",
        state_file_path=state_file,
        threshold=1,
        return_details=True
    )

    assert isinstance(result, dict)
    assert result["is_unseen"] is True
    assert result["mapped_pattern"] == "abc"
    assert result["prediction"] == 0