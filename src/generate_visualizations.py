from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


FIGURE_DIR = Path("results/figures")


def ensure_dirs():
    folders = [
        "heatmap",
        "parameter_analysis",
        "state_diagram",
    ]

    for folder in folders:
        (FIGURE_DIR / folder).mkdir(parents=True, exist_ok=True)


def load_transition_matrix():
    path = Path("results/transition_probabilities/SKAB/fold_1/train.csv")

    if not path.exists():
        raise FileNotFoundError(f"Transition probability file not found: {path}")

    matrix = pd.read_csv(path, index_col=0)
    return matrix


def plot_transition_probability_heatmap():
    matrix = load_transition_matrix()

    plt.figure(figsize=(12, 10))
    plt.imshow(matrix.values)

    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(matrix.index)), matrix.index, fontsize=6)

    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.title("Transition Probability Heatmap - SKAB Fold 1 Train")

    plt.colorbar(label="Transition Probability")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "heatmap" / "transition_probability_heatmap.png", dpi=300)
    plt.close()


def plot_parameter_sensitivity():
    window_path = Path("results/parameter_analysis/window_size_summary.csv")
    alphabet_path = Path("results/parameter_analysis/alphabet_size_summary.csv")

    if not window_path.exists():
        raise FileNotFoundError(f"Window summary file not found: {window_path}")

    if not alphabet_path.exists():
        raise FileNotFoundError(f"Alphabet summary file not found: {alphabet_path}")

    window_df = pd.read_csv(window_path)
    alphabet_df = pd.read_csv(alphabet_path)

    for dataset in window_df["dataset"].unique():
        dataset_df = window_df[window_df["dataset"] == dataset]

        plt.figure(figsize=(7, 4))
        plt.plot(dataset_df["window_size"], dataset_df["f1_mean"], marker="o")
        plt.xlabel("Window Size")
        plt.ylabel("Mean F1-score")
        plt.title(f"Window Size Sensitivity - {dataset}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / "parameter_analysis" / f"window_size_f1_sensitivity_{dataset}.png",
            dpi=300
        )
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(dataset_df["window_size"], dataset_df["state_count_mean"], marker="o")
        plt.xlabel("Window Size")
        plt.ylabel("Mean State Count")
        plt.title(f"Window Size vs State Count - {dataset}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / "parameter_analysis" / f"window_size_state_count_{dataset}.png",
            dpi=300
        )
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(dataset_df["window_size"], dataset_df["transition_density_mean"], marker="o")
        plt.xlabel("Window Size")
        plt.ylabel("Mean Transition Density")
        plt.title(f"Window Size vs Transition Density - {dataset}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / "parameter_analysis" / f"window_size_transition_density_{dataset}.png",
            dpi=300
        )
        plt.close()

    for dataset in alphabet_df["dataset"].unique():
        dataset_df = alphabet_df[alphabet_df["dataset"] == dataset]

        plt.figure(figsize=(7, 4))
        plt.plot(dataset_df["alphabet_size"], dataset_df["f1_mean"], marker="o")
        plt.xlabel("Alphabet Size")
        plt.ylabel("Mean F1-score")
        plt.title(f"Alphabet Size Sensitivity - {dataset}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / "parameter_analysis" / f"alphabet_size_f1_sensitivity_{dataset}.png",
            dpi=300
        )
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(dataset_df["alphabet_size"], dataset_df["state_count_mean"], marker="o")
        plt.xlabel("Alphabet Size")
        plt.ylabel("Mean State Count")
        plt.title(f"Alphabet Size vs State Count - {dataset}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / "parameter_analysis" / f"alphabet_size_state_count_{dataset}.png",
            dpi=300
        )
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(dataset_df["alphabet_size"], dataset_df["transition_density_mean"], marker="o")
        plt.xlabel("Alphabet Size")
        plt.ylabel("Mean Transition Density")
        plt.title(f"Alphabet Size vs Transition Density - {dataset}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / "parameter_analysis" / f"alphabet_size_transition_density_{dataset}.png",
            dpi=300
        )
        plt.close()


def plot_automata_state_diagram(top_n=20):
    matrix = load_transition_matrix()

    edges = []

    for from_state in matrix.index:
        for to_state in matrix.columns:
            probability = float(matrix.loc[from_state, to_state])

            if from_state != to_state and probability > 0:
                edges.append((from_state, to_state, probability))

    edges = sorted(edges, key=lambda item: item[2], reverse=True)[:top_n]

    if not edges:
        raise ValueError("State diagram için çizilecek transition bulunamadı.")

    graph = nx.DiGraph()

    for from_state, to_state, probability in edges:
        graph.add_edge(from_state, to_state, weight=probability)

    plt.figure(figsize=(12, 9))

    pos = nx.spring_layout(
        graph,
        seed=42,
        k=1.5,
        iterations=100
    )

    edge_weights = [
        graph[u][v]["weight"] * 4
        for u, v in graph.edges()
    ]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1800,
        edgecolors="black",
        linewidths=1.5
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=9,
        font_weight="bold"
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=edge_weights,
        connectionstyle="arc3,rad=0.12"
    )

    edge_labels = {
        (u, v): f"{data['weight']:.2f}"
        for u, v, data in graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.55
    )

    plt.title("Automata State Diagram - Top Transition Probabilities")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "state_diagram" / "automata_state_diagram.png", dpi=300)
    plt.close()

def plot_confusion_matrix_from_predictions():
    prediction_path = Path("results/predictions/automata_prediction_outputs.csv")

    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction output file not found: {prediction_path}")

    df = pd.read_csv(prediction_path)

    y_true = df["y_true"].astype(int).tolist()
    y_pred = df["y_pred"].astype(int).tolist()

    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)

    matrix = [
        [tn, fp],
        [fn, tp]
    ]

    output_dir = FIGURE_DIR / "confusion_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Normal", "Anomaly"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix)

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - Automata Predictions")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def plot_precision_recall_curve_from_predictions():
    prediction_path = Path("results/predictions/automata_prediction_outputs.csv")

    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction output file not found: {prediction_path}")

    df = pd.read_csv(prediction_path)

    y_true = df["y_true"].astype(int).tolist()
    scores = df["anomaly_score"].astype(float).tolist()

    thresholds = sorted(set(scores), reverse=True)

    precision_values = []
    recall_values = []

    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in scores]

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        precision_values.append(precision)
        recall_values.append(recall)

    output_dir = FIGURE_DIR / "pr_curve"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(recall_values, precision_values, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Automata Predictions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=300)
    plt.close()

def main():
    ensure_dirs()
    plot_transition_probability_heatmap()
    plot_parameter_sensitivity()
    plot_automata_state_diagram()
    plot_confusion_matrix_from_predictions()
    plot_precision_recall_curve_from_predictions()

    print("Visualizations generated from real result CSV files.")


if __name__ == "__main__":
    main()