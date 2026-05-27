import time
import csv
from pathlib import Path
import subprocess


RESULTS_DIR = Path("results/runtime")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


COMMANDS = [
    {
        "model": "Automata",
        "task": "prediction_output",
        "command": ["python3", "-m", "src.generate_automata_prediction_outputs"]
    },
    {
        "model": "Automata",
        "task": "visualization",
        "command": ["python3", "-m", "src.generate_visualizations"]
    },
    {
        "model": "Explainability",
        "task": "unit_test",
        "command": ["python3", "-m", "pytest", "tests/test_explainability.py"]
    }
]


def measure_command(command):
    start_time = time.perf_counter()

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    end_time = time.perf_counter()

    return {
        "runtime_seconds": end_time - start_time,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip()
    }


def main():
    rows = []

    for item in COMMANDS:
        print(f"Ölçülüyor: {item['model']} - {item['task']}")

        result = measure_command(item["command"])

        rows.append({
            "model": item["model"],
            "task": item["task"],
            "runtime_seconds": round(result["runtime_seconds"], 4),
            "return_code": result["return_code"]
        })

        if result["return_code"] != 0:
            print("Hata oluştu:")
            print(result["stderr"])

    output_path = RESULTS_DIR / "runtime_summary.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
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

    print(f"Runtime summary saved to: {output_path}")


if __name__ == "__main__":
    main()
