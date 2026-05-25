import os
import pandas as pd

RESULTS_DIR = "results/unseen"

files = [
    f"{RESULTS_DIR}/skab_unseen_deep_learning_summary.csv",
    f"{RESULTS_DIR}/skab_unseen_automata_summary.csv",
]

rows = []

for file in files:
    df = pd.read_csv(file)
    pivot = df.pivot_table(
        index=["dataset", "model", "scenario"],
        columns="metric",
        values="mean"
    ).reset_index()
    rows.append(pivot)

comparison = pd.concat(rows, ignore_index=True)
comparison = comparison[["dataset", "model", "scenario", "accuracy", "precision", "recall", "f1"]]

os.makedirs(RESULTS_DIR, exist_ok=True)
comparison.to_csv(f"{RESULTS_DIR}/skab_unseen_model_comparison.csv", index=False)

print(comparison)
print(f"\nKaydedildi: {RESULTS_DIR}/skab_unseen_model_comparison.csv")
