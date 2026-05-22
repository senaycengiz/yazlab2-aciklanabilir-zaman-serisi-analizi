from src.automata_predict import predict_sequence


def test_automata_prediction_accuracy():

    known_patterns = [
        "abc",
        "bca",
        "cab"
    ]

    test_data = [
        ("abc", 0),
        ("abd", 0),
        ("bca", 0),
        ("zzz", 1),
        ("yyy", 1)
    ]

    correct_count = 0

    for sequence, true_label in test_data:
        prediction = predict_sequence(
            sequence=sequence,
            known_patterns=known_patterns,
            threshold=1
        )

        if prediction == true_label:
            correct_count += 1

    accuracy = correct_count / len(test_data)

    assert accuracy >= 0.80
