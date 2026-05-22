import pandas as pd

from src.automata_predict import predict_from_state_file


def test_predict_from_state_file(tmp_path):

    state_file = tmp_path / "states.csv"

    df = pd.DataFrame({
        "state": ["q0", "q1", "q2"],
        "pattern": ["abc", "bca", "cab"]
    })

    df.to_csv(state_file, index=False)

    known_prediction = predict_from_state_file(
        sequence="abc",
        state_file_path=state_file,
        threshold=1
    )

    anomaly_prediction = predict_from_state_file(
        sequence="zzz",
        state_file_path=state_file,
        threshold=1
    )

    assert known_prediction == 0
    assert anomaly_prediction == 1
