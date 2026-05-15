import os
import pandas as pd
import matplotlib.pyplot as plt


def save_history_csv(history, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(output_path, index=False)


def save_loss_plot(history, output_path, title):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Training Loss")
    plt.plot(history["epoch"], history["validation_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
