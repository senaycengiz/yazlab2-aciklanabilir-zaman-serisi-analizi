from src.pattern_analysis import analyze_pattern_counts_for_values


def test_analyze_pattern_counts_for_values():
    sax_values = ["a", "b", "c", "d", "e"]

    results = analyze_pattern_counts_for_values(sax_values)

    window_sizes = [row["window_size"] for row in results]

    assert 3 in window_sizes
    assert 4 in window_sizes
    assert 5 in window_sizes
    assert 6 in window_sizes