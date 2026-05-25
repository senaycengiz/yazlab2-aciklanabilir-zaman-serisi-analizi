import os
import pandas as pd


RESULT_DIR = "results/unseen"
AUTOMATA_STATE_DIR = "data/processed/automata/states"


def mutate_pattern(pattern: str, alphabet: list[str]) -> str:
    """
    Var olan pattern'i küçük bir değişiklikle yeni pattern'e dönüştürür.
    Amaç: train sözlüğünde olmayan kontrollü unseen pattern üretmek.
    """
    chars = list(pattern)

    for i in range(len(chars) - 1, -1, -1):
        current = chars[i]
        for symbol in alphabet:
            if symbol != current:
                chars[i] = symbol
                return "".join(chars)

    return pattern


def create_unseen_file(train_path: str, test_path: str, output_path: str) -> dict:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_patterns = set(train_df["pattern"].astype(str))
    test_df["original_pattern"] = test_df["pattern"].astype(str)

    alphabet = sorted(set("".join(train_patterns)))

    unseen_patterns = []
    is_unseen_values = []

    for pattern in test_df["original_pattern"]:
        new_pattern = mutate_pattern(pattern, alphabet)

        attempt = 0
        while new_pattern in train_patterns and attempt < len(pattern):
            rotated = pattern[attempt:] + pattern[:attempt]
            new_pattern = mutate_pattern(rotated, alphabet)
            attempt += 1

        if new_pattern in train_patterns:
            new_pattern = pattern + alphabet[-1]

        unseen_patterns.append(new_pattern)
        is_unseen_values.append(new_pattern not in train_patterns)

    test_df["pattern"] = unseen_patterns
    test_df["is_unseen"] = is_unseen_values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_df.to_csv(output_path, index=False)

    return {
        "output": output_path,
        "total_rows": len(test_df),
        "unseen_rows": int(test_df["is_unseen"].sum()),
        "unseen_ratio": round(float(test_df["is_unseen"].mean()), 4)
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    summaries = []

    # SKAB fold bazlı unseen üretimi
    for fold in range(1, 6):
        train_path = f"{AUTOMATA_STATE_DIR}/SKAB/fold_{fold}/train.csv"
        test_path = f"{AUTOMATA_STATE_DIR}/SKAB/fold_{fold}/test.csv"
        output_path = f"{RESULT_DIR}/skab_fold_{fold}_unseen.csv"

        if os.path.exists(train_path) and os.path.exists(test_path):
            summaries.append(create_unseen_file(train_path, test_path, output_path))

    # BATADAL zaman sıralı test için unseen üretimi
    batadal_train = f"{AUTOMATA_STATE_DIR}/BATADAL/BATADAL_dataset04/train.csv"
    batadal_test = f"{AUTOMATA_STATE_DIR}/BATADAL/BATADAL_dataset04/test.csv"
    batadal_output = f"{RESULT_DIR}/batadal_unseen.csv"

    if os.path.exists(batadal_train) and os.path.exists(batadal_test):
        summaries.append(create_unseen_file(batadal_train, batadal_test, batadal_output))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(f"{RESULT_DIR}/unseen_dataset_summary.csv", index=False)

    print("Unseen veri dosyaları oluşturuldu.")
    print(summary_df)


if __name__ == "__main__":
    main()
