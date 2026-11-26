import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import read_json
from settings import ARTIFACTS, RUNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["scratch", "transfer"])
    args = parser.parse_args()

    history = read_json(
        ARTIFACTS / f"{args.kind}_history.json"
    )

    rows = [
        row
        for row in history
        if row["epoch"] > 0
    ]

    expected_epochs = RUNS[args.kind]["epochs"]
    if not rows or rows[-1]["epoch"] != expected_epochs:
        raise RuntimeError(
            "Полное обучение ещё не завершено"
        )

    epochs = [row["epoch"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epochs,
        [row["train_loss"] for row in rows],
        marker="o",
        label="Train",
    )
    ax.plot(
        epochs,
        [row["val_loss"] for row in rows],
        marker="o",
        label="Validation",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy")
    ax.set_title(f"{args.kind}: training and validation loss")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        ARTIFACTS / f"{args.kind}_loss.png",
        dpi=180,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epochs,
        [row["val_perplexity"] for row in rows],
        marker="o",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation perplexity (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"{args.kind}: validation perplexity")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        ARTIFACTS / f"{args.kind}_perplexity.png",
        dpi=180,
    )
    plt.close(fig)

    selected = sorted({
        1,
        (expected_epochs + 1) // 2,
        expected_epochs,
    })

    text = []

    for epoch in selected:
        row = next(
            row
            for row in rows
            if row["epoch"] == epoch
        )

        sample = row["sample"]

        text.extend([
            f"EPOCH {epoch}",
            f"Prompt: {sample['prompt']}",
            "Strategy: temperature, temperature=0.8, seed=42",
            f"Generated tokens: {sample['new_tokens']}",
            f"Validation PPL: {row['val_perplexity']:.4f}",
            "Continuation:",
            sample["continuation"],
            "",
        ])

    path = ARTIFACTS / f"{args.kind}_progress.txt"
    path.write_text("\n".join(text), encoding="utf-8")

    print("Графики сохранены в:", ARTIFACTS)
    print("Примеры генерации:", path)
    print("PLOTS_OK")


if __name__ == "__main__":
    main()