import os
import pandas as pd


PATTERN_DATA_DIR = "data/processed/patterns"
AUTOMATA_STATE_DIR = "data/processed/automata/states"
AUTOMATA_TRANSITION_DIR = "data/processed/automata/transitions"


def create_states(patterns):
    """
    Benzersiz patternleri automata state olarak tanımlar.
    Her benzersiz pattern q0, q1, q2... şeklinde isimlendirilir.
    """
    unique_patterns = sorted(set(patterns))

    state_map = {
        pattern: f"q{index}" for index, pattern in enumerate(unique_patterns)
    }

    return state_map


def create_transition_list(patterns, state_map):
    """
    Pattern sırasına göre state geçişlerini oluşturur.
    """
    transitions = []

    for i in range(len(patterns) - 1):
        from_pattern = patterns[i]
        to_pattern = patterns[i + 1]

        transitions.append({
            "from_state": state_map[from_pattern],
            "to_state": state_map[to_pattern],
            "from_pattern": from_pattern,
            "to_pattern": to_pattern
        })

    return transitions


def build_automata(input_path, state_output_path, transition_output_path):
    """
    Pattern dosyasından state listesi ve transition listesi üretir.
    """
    df = pd.read_csv(input_path)

    if "pattern" not in df.columns:
        raise ValueError(f"{input_path} dosyasında pattern kolonu bulunamadı.")

    patterns = df["pattern"].astype(str).tolist()

    state_map = create_states(patterns)
    transitions = create_transition_list(patterns, state_map)

    states_df = pd.DataFrame([
        {"state": state, "pattern": pattern}
        for pattern, state in state_map.items()
    ])

    transitions_df = pd.DataFrame(transitions)

    os.makedirs(os.path.dirname(state_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(transition_output_path), exist_ok=True)

    states_df.to_csv(state_output_path, index=False)
    transitions_df.to_csv(transition_output_path, index=False)

    print(f"State sayısı: {len(states_df)}")
    print(f"Transition sayısı: {len(transitions_df)}")
    print(f"State dosyası: {state_output_path}")
    print(f"Transition dosyası: {transition_output_path}")


def process_all_pattern_files():
    """
    Tüm pattern dosyaları için automata state ve transition listeleri üretir.
    """
    if not os.path.exists(PATTERN_DATA_DIR):
        print(f"Atlandı, pattern klasörü yok: {PATTERN_DATA_DIR}")
        return

    for root, dirs, files in os.walk(PATTERN_DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(input_path, PATTERN_DATA_DIR)

                state_output_path = os.path.join(AUTOMATA_STATE_DIR, relative_path)
                transition_output_path = os.path.join(AUTOMATA_TRANSITION_DIR, relative_path)

                build_automata(
                    input_path=input_path,
                    state_output_path=state_output_path,
                    transition_output_path=transition_output_path
                )


if __name__ == "__main__":
    process_all_pattern_files()