from src.unseen_pattern_checker import is_seen_pattern


def test_seen_pattern():

    sax_dictionary = {
        "a": 0,
        "b": 1,
        "c": 2
    }

    assert is_seen_pattern("a", sax_dictionary) is True


def test_unseen_pattern():

    sax_dictionary = {
        "a": 0,
        "b": 1,
        "c": 2
    }

    assert is_seen_pattern("d", sax_dictionary) is False