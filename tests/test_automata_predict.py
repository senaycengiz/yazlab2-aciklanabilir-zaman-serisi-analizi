from src.automata_predict import predict_sequence


def test_known_pattern():

    automata = {
        "abc": {},
        "bca": {},
        "cab": {}
    }

    prediction = predict_sequence(
        "abc",
        automata
    )

    assert prediction == 0


def test_unknown_pattern():

    automata = {
        "abc": {},
        "bca": {},
        "cab": {}
    }

    prediction = predict_sequence(
        "zzz",
        automata
    )

    assert prediction == 1
