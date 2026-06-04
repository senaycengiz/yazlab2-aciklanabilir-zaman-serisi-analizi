import time
import csv
import subprocess
from pathlib import Path


RESULTS_DIR = Path("results/runtime")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


COMMANDS = [
    {
        "model": "LSTM",
        "task": "training",
        "command": ["python3", "-m", "src.train_lstm"]
    },
    {
        "model": "GRU",
        "task": "training",
        "command": ["python3", "-m", "src.train_gru"]
    },
    {
        "model": "1D-CNN",
        "task": "training",
        "command": ["python3", "-m", "src.train_cnn"]
    },
    {
        "model": "Automata",
        "task": "inference",
        "command": [
            "python3",
            "-m",
            "src.generate_automata_prediction_outputs"
        ]
    }
]


def measure_runtime(command):
    start = time.perf_counter()

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    end = time.perf_counter()

    return {
        "runtime": round(end - start, 4),
        "return_code": result.returncode
    }


def main():
    rows = []

    for item in COMMANDS:
        print(f"Çalıştırılıyor: {item['model']}")

        result = measure_runtime(item["command"])

        rows.append({
            "model": item["model"],
            "task": item["task"],
            "runtime_seconds": result["runtime"],
            "return_code": result["return_code"]
        })

    output_path = RESULTS_DIR / "model_runtime_results.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "task",
                "runtime_seconds",
                "return_code"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nKaydedildi: {output_path}")


if __name__ == "__main__":
    main()
