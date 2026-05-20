from src.transition_analysis import calculate_transition_counts, create_transition_matrix


def test_calculate_transition_counts():
    patterns = ["aaa", "aab", "aaa", "aab"]

    transitions = calculate_transition_counts(patterns)

    assert transitions[("aaa", "aab")] == 2
    assert transitions[("aab", "aaa")] == 1


def test_create_transition_matrix():
    patterns = ["aaa", "aab", "aaa"]

    matrix = create_transition_matrix(patterns)

    assert matrix.loc["aaa", "aab"] == 1
    assert matrix.loc["aab", "aaa"] == 1