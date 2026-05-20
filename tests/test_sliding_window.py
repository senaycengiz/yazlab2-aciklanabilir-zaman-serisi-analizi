from src.sliding_window import create_patterns


def test_create_patterns():
    sax_values = ["a", "b", "c", "d", "e"]

    patterns = create_patterns(sax_values, window_size=3)

    assert patterns == ["abc", "bcd", "cde"]


def test_create_patterns_when_window_larger_than_data():
    sax_values = ["a", "b"]

    patterns = create_patterns(sax_values, window_size=3)

    assert patterns == []