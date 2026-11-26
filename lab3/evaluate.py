import argparse
import csv
import re
import statistics

from torch.utils.data import DataLoader

from common import (
    TokenWindows,
    evaluate_loss,
    generate,
    get_device,
    get_metadata,
    load_best_model,
    load_tokenizer,
    load_tokens,
    read_json,
    write_json,
)
from settings import (
    ARTIFACTS,
    BATCH_SIZE,
    CONTEXT,
    MAX_NEW_TOKENS,
    MODEL_ID,
    PROMPTS,
    SAMPLING_SEEDS,
    STRATEGIES,
)


def distinct(words, n):
    ngrams = [
        tuple(words[index:index + n])
        for index in range(len(words) - n + 1)
    ]

    if not ngrams:
        return None

    return len(set(ngrams)) / len(ngrams)


def text_metrics(text):
    words = re.findall(r"\b\w+\b", text.lower())

    d1 = distinct(words, 1)
    d2 = distinct(words, 2)

    return {
        "words": len(words),
        "distinct_1": d1,
        "distinct_2": d2,
        "repeat_2": None if d2 is None else 1.0 - d2,
    }


def summarize(values):
    valid = [
        value
        for value in values
        if value is not None
    ]

    if not valid:
        return {
            "mean": None,
            "std": None,
            "valid_samples": 0,
        }

    return {
        "mean": statistics.mean(valid),
        "std": statistics.pstdev(valid),
        "valid_samples": len(valid),
    }


def formatted(value):
    return "n/a" if value is None else f"{value:.4f}"


def evaluate_model(kind):
    device = get_device()
    model, checkpoint = load_best_model(kind, device)
    tokenizer = load_tokenizer()
    metadata = get_metadata()

    loader = DataLoader(
        TokenWindows(load_tokens("validation")),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    validation = evaluate_loss(model, loader, device)

    samples = []
    aggregates = {}

    for strategy in STRATEGIES:
        seeds = (
            [SAMPLING_SEEDS[0]]
            if strategy == "greedy"
            else SAMPLING_SEEDS
        )

        group = []

        for prompt in PROMPTS:
            for seed in seeds:
                sample = generate(
                    model,
                    tokenizer,
                    prompt,
                    strategy,
                    seed,
                    device,
                )

                sample["metrics"] = text_metrics(
                    sample["continuation"]
                )

                samples.append(sample)
                group.append(sample)

        aggregates[strategy] = {
            "samples": len(group),
            "new_tokens": summarize([
                sample["new_tokens"]
                for sample in group
            ]),
            **{
                name: summarize([
                    sample["metrics"][name]
                    for sample in group
                ])
                for name in (
                    "words",
                    "distinct_1",
                    "distinct_2",
                    "repeat_2",
                )
            },
        }

    protocol = {
        "tokenizer": MODEL_ID,
        "context": CONTEXT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompts": PROMPTS,
        "seeds": SAMPLING_SEEDS,
        "strategies": STRATEGIES,
        "data_hashes": {
            split: metadata[split]["tokens_sha256"]
            for split in ("train", "validation")
        },
        "ngram_units": "lowercase words without punctuation",
    }

    report = {
        "kind": kind,
        "checkpoint_epoch": checkpoint["epoch"],
        "validation": validation,
        "protocol": protocol,
        "strategies": aggregates,
        "samples": samples,
    }

    write_json(
        ARTIFACTS / f"{kind}_evaluation.json",
        report,
    )

    columns = [
        "strategy",
        "samples",
        "mean_new_tokens",
        "distinct_1_mean",
        "distinct_1_std",
        "distinct_2_mean",
        "distinct_2_std",
        "repeat_2_mean",
        "valid_distinct_2_samples",
    ]

    with (
        ARTIFACTS / f"{kind}_decoding.csv"
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for strategy, values in aggregates.items():
            writer.writerow({
                "strategy": strategy,
                "samples": values["samples"],
                "mean_new_tokens": values["new_tokens"]["mean"],
                "distinct_1_mean": values["distinct_1"]["mean"],
                "distinct_1_std": values["distinct_1"]["std"],
                "distinct_2_mean": values["distinct_2"]["mean"],
                "distinct_2_std": values["distinct_2"]["std"],
                "repeat_2_mean": values["repeat_2"]["mean"],
                "valid_distinct_2_samples": (
                    values["distinct_2"]["valid_samples"]
                ),
            })

    full_text = []

    for sample in samples:
        full_text.extend([
            f"STRATEGY: {sample['strategy']}",
            f"SEED: {sample['seed']}",
            f"PROMPT: {sample['prompt']}",
            f"NEW TOKENS: {sample['new_tokens']}",
            "CONTINUATION:",
            sample["continuation"],
            "",
        ])

    (
        ARTIFACTS / f"{kind}_generations.txt"
    ).write_text("\n".join(full_text), encoding="utf-8")

    representative = [
        sample
        for sample in samples
        if sample["prompt"] == PROMPTS[0]
        and sample["seed"] == SAMPLING_SEEDS[0]
    ]

    (
        ARTIFACTS / f"{kind}_examples.txt"
    ).write_text(
        "\n\n".join(
            f"{sample['strategy']}\n"
            f"Prompt: {sample['prompt']}\n"
            f"{sample['continuation']}"
            for sample in representative
        ),
        encoding="utf-8",
    )

    print("Модель:", kind)
    print("Эпоха checkpoint:", checkpoint["epoch"])
    print("Целевых токенов:", validation["target_tokens"])
    print(f"Validation loss: {validation['loss']:.6f}")
    print(f"Validation PPL: {validation['perplexity']:.4f}")
    print("strategy | samples | distinct-1 | distinct-2 | repeat-2")

    for strategy, values in aggregates.items():
        print(
            f"{strategy} | "
            f"{values['samples']} | "
            f"{formatted(values['distinct_1']['mean'])} | "
            f"{formatted(values['distinct_2']['mean'])} | "
            f"{formatted(values['repeat_2']['mean'])}"
        )

    print("EVALUATION_OK")


def compare_models():
    reports = {
        kind: read_json(
            ARTIFACTS / f"{kind}_evaluation.json"
        )
        for kind in ("scratch", "transfer")
    }

    if reports["scratch"]["protocol"] != reports["transfer"]["protocol"]:
        raise RuntimeError("Протоколы оценки моделей отличаются")

    rows = []

    for kind in ("scratch", "transfer"):
        summary = read_json(
            ARTIFACTS / f"{kind}_summary.json"
        )
        evaluation = reports[kind]

        rows.append({
            "model": kind,
            "best_epoch": evaluation["checkpoint_epoch"],
            "parameters": summary["trainable_parameters"],
            "val_loss": evaluation["validation"]["loss"],
            "val_perplexity": (
                evaluation["validation"]["perplexity"]
            ),
            "training_minutes": summary["training_seconds"] / 60,
            "validation_minutes": summary["validation_seconds"] / 60,
            "peak_process_rss_gib": summary["peak_process_rss_gib"],
            "sampled_peak_mps_driver_gib": (
                summary["sampled_peak_mps_driver_gib"]
            ),
        })

    with (
        ARTIFACTS / "comparison.csv"
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    write_json(ARTIFACTS / "comparison.json", rows)

    print("model | epoch | parameters | PPL | train_min | MPS_GiB")

    for row in rows:
        print(
            f"{row['model']} | "
            f"{row['best_epoch']} | "
            f"{row['parameters']} | "
            f"{row['val_perplexity']:.3f} | "
            f"{row['training_minutes']:.2f} | "
            f"{row['sampled_peak_mps_driver_gib']:.3f}"
        )

    examples = []

    for kind in ("scratch", "transfer"):
        sample = next(
            sample
            for sample in reports[kind]["samples"]
            if sample["prompt"] == PROMPTS[0]
            and sample["strategy"] == "temperature"
            and sample["seed"] == SAMPLING_SEEDS[0]
        )

        examples.extend([
            kind.upper(),
            f"Prompt: {sample['prompt']}",
            "Temperature: 0.8; seed: 42",
            sample["continuation"],
            "",
        ])

    (
        ARTIFACTS / "comparison_examples.txt"
    ).write_text("\n".join(examples), encoding="utf-8")

    print("COMPARISON_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind",
        choices=["scratch", "transfer", "compare"],
    )
    args = parser.parse_args()

    if args.kind == "compare":
        compare_models()
    else:
        evaluate_model(args.kind)


if __name__ == "__main__":
    main()