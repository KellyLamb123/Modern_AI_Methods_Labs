import hashlib
import json
import re

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from settings import (
    ARTIFACTS,
    CHECKPOINT_DIR,
    CONTEXT,
    DATA_DIR,
    DATASET_CONFIG,
    DATASET_ID,
    MODEL_ID,
)


def article_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_articles(rows):
    articles = []
    buffer = []

    for row in rows:
        line = row["text"].rstrip("\n")

        # Один знак "=" с каждой стороны означает заголовок статьи.
        # Подразделы вида "= = Gameplay = =" сюда не попадают.
        is_article_title = re.fullmatch(
            r"=\s+[^=]+?\s+=",
            line.strip(),
        ) is not None

        if is_article_title and buffer:
            text = "\n".join(buffer).strip()
            if text:
                articles.append(text)
            buffer = []

        if buffer or line.strip():
            buffer.append(line)

    text = "\n".join(buffer).strip()
    if text:
        articles.append(text)

    return articles


def main():
    if CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.glob("*.pt")):
        raise RuntimeError(
            "Найдены checkpoint моделей. "
            "Не пересоздавайте данные после начала обучения."
        )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )

    assert tokenizer.eos_token_id is not None

    articles = {
        split: collect_articles(dataset[split])
        for split in ("train", "validation")
    }

    train_hashes = {
        article_hash(text)
        for text in articles["train"]
    }
    validation_hashes = {
        article_hash(text)
        for text in articles["validation"]
    }

    overlap = train_hashes & validation_hashes
    if overlap:
        raise RuntimeError(
            f"Обнаружено одинаковых статей между частями: {len(overlap)}"
        )

    metadata = {
        "dataset": DATASET_ID,
        "configuration": DATASET_CONFIG,
        "tokenizer": MODEL_ID,
        "vocab_size": len(tokenizer),
        "context": CONTEXT,
        "exact_article_overlap": len(overlap),
        "source_rows": {
            split: len(dataset[split])
            for split in ("train", "validation", "test")
        },
    }

    for split in ("train", "validation"):
        encoded = tokenizer(
            articles[split],
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
            verbose=False,
        )["input_ids"]

        token_ids = []
        for ids in encoded:
            token_ids.extend(ids)
            token_ids.append(tokenizer.eos_token_id)

        tensor = torch.tensor(token_ids, dtype=torch.long)
        windows = (len(tensor) - 1) // CONTEXT

        if windows < 1:
            raise RuntimeError(f"Недостаточно данных в части {split}")

        torch.save(tensor, DATA_DIR / f"{split}.pt")

        metadata[split] = {
            "articles": len(articles[split]),
            "tokens": len(tensor),
            "windows": windows,
            "evaluated_target_tokens": windows * CONTEXT,
            "unused_tail_target_tokens": (
                len(tensor) - 1 - windows * CONTEXT
            ),
            "tokens_sha256": hashlib.sha256(
                tensor.numpy().tobytes()
            ).hexdigest(),
        }

    tokenizer.save_pretrained(DATA_DIR / "tokenizer")

    (ARTIFACTS / "data_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_tokens = torch.load(
        DATA_DIR / "train.pt",
        weights_only=True,
        map_location="cpu",
    )

    x = train_tokens[:CONTEXT]
    y = train_tokens[1:CONTEXT + 1]

    assert torch.equal(x[1:], y[:-1])
    assert len(x) == len(y) == CONTEXT
    assert train_tokens.min().item() >= 0
    assert train_tokens.max().item() < len(tokenizer)

    print("Датасет:", DATASET_ID)
    print("Конфигурация:", DATASET_CONFIG)
    print("Исходные строки:", metadata["source_rows"])
    print("Токенизатор:", type(tokenizer).__name__)
    print("Размер словаря:", len(tokenizer))
    print("Окно контекста:", CONTEXT)
    print("Совпадающих статей train/validation:", len(overlap))

    for split in ("train", "validation"):
        print(split, metadata[split])

    print("Форма одного входа:", tuple(x.shape))
    print("Форма одной цели:", tuple(y.shape))
    print("Первые 10 входных ID:", x[:10].tolist())
    print("Первые 10 целевых ID:", y[:10].tolist())
    print("Первые токены:", tokenizer.convert_ids_to_tokens(x[:10].tolist()))
    print("DATA_PREPARATION_OK")


if __name__ == "__main__":
    main()