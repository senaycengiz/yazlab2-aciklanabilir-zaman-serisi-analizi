from src.explainability import (
    levenshtein_distance,
    find_nearest_pattern,
    calculate_transition_probability,
    calculate_path_probability,
    build_explanation
)


def test_levenshtein_distance_same_pattern():
    assert levenshtein_distance("abc", "abc") == 0


def test_levenshtein_distance_different_pattern():
    assert levenshtein_distance("adc", "abc") == 1


def test_find_nearest_pattern():
    known_patterns = ["aaa", "abc", "bcc"]
    nearest_pattern, distance = find_nearest_pattern("adc", known_patterns)

    assert nearest_pattern == "abc"
    assert distance == 1


def test_transition_probability():
    transition_counts = {
        "aab": {
            "abc": 72,
            "bcc": 28
        }
    }

    probability = calculate_transition_probability(
        from_state="aab",
        to_state="abc",
        transition_counts=transition_counts
    )

    assert probability == 0.72


def test_path_probability():
    transition_counts = {
        "aab": {
            "abc": 72,
            "bcc": 28
        },
        "abc": {
            "bcc": 15,
            "aaa": 85
        }
    }

    states = ["aab", "abc", "bcc"]

    path_probability, transitions = calculate_path_probability(
        states=states,
        transition_counts=transition_counts
    )

    assert round(path_probability, 3) == 0.108
    assert len(transitions) == 2


def test_build_explanation_for_unseen_pattern():
    known_patterns = ["aab", "abc", "bcc"]

    transition_counts = {
        "aab": {
            "abc": 72,
            "bcc": 28
        },
        "abc": {
            "bcc": 15,
            "aaa": 85
        }
    }

    explanation = build_explanation(
        time_step=5,
        previous_state="aab",
        incoming_pattern="adc",
        known_patterns=known_patterns,
        transition_counts=transition_counts,
        next_state="bcc",
        anomaly_threshold=0.20
    )

    assert explanation["time_step"] == 5
    assert explanation["state"] == "aab"
    assert explanation["pattern"] == "adc"
    assert explanation["status"] == "unseen"
    assert explanation["mapped_to"] == "abc"
    assert explanation["levenshtein_distance"] == 1
    assert round(explanation["path_probability"], 3) == 0.108
    assert explanation["decision"] == "anomaly"
    assert explanation["confidence_score"] == explanation["path_probability"]


def test_build_explanation_for_seen_pattern():
    known_patterns = ["aab", "abc", "bcc"]

    transition_counts = {
        "aab": {
            "abc": 90,
            "bcc": 10
        },
        "abc": {
            "bcc": 80,
            "aaa": 20
        }
    }

    explanation = build_explanation(
        time_step=10,
        previous_state="aab",
        incoming_pattern="abc",
        known_patterns=known_patterns,
        transition_counts=transition_counts,
        next_state="bcc",
        anomaly_threshold=0.20
    )

    assert explanation["status"] == "seen"
    assert explanation["mapped_to"] == "abc"
    assert round(explanation["path_probability"], 2) == 0.72
    assert explanation["decision"] == "normal"
