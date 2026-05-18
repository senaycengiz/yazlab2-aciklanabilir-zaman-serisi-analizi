import pandas as pd
from src.evaluation.metrics import calculate_metrics


def test_calculate_metrics():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    metrics = calculate_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics

    assert metrics["accuracy"] >= 0
    assert metrics["accuracy"] <= 1