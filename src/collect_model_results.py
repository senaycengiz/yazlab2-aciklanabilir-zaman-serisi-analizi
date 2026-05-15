import os
import pandas as pd


RESULTS_ROOT = "results"
OUTPUT_PATH = "results/all_model_results.csv"


def collect_results():
    all_results = []

    model_names = ["lstm", "gru", "cnn"]

    for model_name in model_names:
        model_dir = os.path.join(RESULTS_ROOT, model_name)

        if not os.path.exists(model_dir):
            continue

        for file_name in os.listdir(model_dir):
            if not file_name.endswith(".csv"):
                continue

            file_path = os.path.join(model_dir, file_name)
            df = pd.read_csv(file_path)

            df["model_group"] = model_name.upper()
            df["source_file"] = file_path

            all_results.append(df)

    if not all_results:
        print("Toplanacak sonuç dosyası bulunamadı.")
        return

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Tüm model sonuçları tek CSV dosyasında toplandı: {OUTPUT_PATH}")
    print(combined_df)


if __name__ == "__main__":
    collect_results()
