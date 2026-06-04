import json
from pathlib import Path

from src.explainability import build_explanation


def main():
    output_dir = Path("results/explainability")
    output_dir.mkdir(parents=True, exist_ok=True)

    known_patterns = ["aab", "abc", "bcc", "aaa"]

    transition_counts = {
        "aab": {
            "abc": 72,
            "bcc": 28
        },
        "abc": {
            "bcc": 15,
            "aaa": 85
        },
        "bcc": {
            "aaa": 60,
            "abc": 40
        }
    }

    explanations = [
        build_explanation(
            time_step=5,
            previous_state="aab",
            incoming_pattern="adc",
            known_patterns=known_patterns,
            transition_counts=transition_counts,
            next_state="bcc",
            anomaly_threshold=0.20
        ),
        build_explanation(
            time_step=10,
            previous_state="aab",
            incoming_pattern="abc",
            known_patterns=known_patterns,
            transition_counts=transition_counts,
            next_state="bcc",
            anomaly_threshold=0.20
        )
    ]

    output_path = output_dir / "sample_explainability_outputs.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(explanations, file, indent=4, ensure_ascii=False)

    print(f"Explainability outputs saved to: {output_path}")


if __name__ == "__main__":
    main()
