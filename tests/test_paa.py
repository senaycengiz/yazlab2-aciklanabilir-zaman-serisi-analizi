from src.paa_transform import apply_paa


def test_paa_segments_are_calculated_correctly():
    series = [1, 2, 3, 4, 5, 6]

    result = apply_paa(series, segment_size=2)

    assert result == [1.5, 3.5, 5.5]


def test_paa_ignores_incomplete_last_segment():
    series = [1, 2, 3, 4, 5]

    result = apply_paa(series, segment_size=2)

    assert result == [1.5, 3.5]