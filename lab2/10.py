import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "task10_output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
MODEL_DIR = OUTPUT_DIR / "model"

MODEL_ID = "distilbert/distilbert-base-uncased"
DATASET_ID = "stanfordnlp/imdb"

SEED = 42
MAX_LENGTH = 256
EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

ID2LABEL = {
    0: "NEGATIVE",
    1: "POSITIVE",
}
LABEL2ID = {
    label: index
    for index, label in ID2LABEL.items()
}

CUSTOM_TEXTS = [
    "The film combines a moving story with excellent acting and beautiful music.",
    "The film wastes an interesting idea on boring scenes and terrible dialogue.",
    "I expected a weak story, but the thoughtful ending made the film worthwhile.",
    "The film is not entertaining, and its impressive visuals cannot save the dull plot.",
]


def compute_metrics(eval_prediction):
    logits = eval_prediction.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    labels = eval_prediction.label_ids
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(
            f1_score(
                labels,
                predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        ),
    }


def print_metrics(title, metrics, prefix):
    print(f"\n{title}")
    print(f"Accuracy: {metrics[f'{prefix}_accuracy']:.6f}")
    print(f"F1-score: {metrics[f'{prefix}_f1']:.6f}")


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить обучение из последнего сохранённого checkpoint.",
    )
    mode.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Оценить сохранённую модель без повторного обучения.",
    )

    arguments = parser.parse_args()

    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if arguments.evaluate_only and not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Сохранённая модель не найдена: {MODEL_DIR}"
        )

    if (
        not arguments.evaluate_only
        and not arguments.resume
        and CHECKPOINT_DIR.exists()
        and any(CHECKPOINT_DIR.glob("checkpoint-*"))
    ):
        raise RuntimeError(
            "Найдены checkpoint предыдущего запуска. "
            "Для продолжения используйте --resume, "
            "для оценки готовой модели — --evaluate-only."
        )

    raw_dataset = load_dataset(DATASET_ID)

    train_validation = raw_dataset["train"].train_test_split(
        test_size=0.2,
        seed=SEED,
        stratify_by_column="label",
    )

    dataset = DatasetDict(
        {
            "train": train_validation["train"],
            "validation": train_validation["test"],
            "test": raw_dataset["test"],
        }
    )

    existing_texts = set(raw_dataset["train"]["text"])
    existing_texts.update(raw_dataset["test"]["text"])

    for text in CUSTOM_TEXTS:
        if text in existing_texts:
            raise ValueError(
                "Собственный текст совпал с примером IMDb."
            )

    del existing_texts

    tokenizer_source = (
        str(MODEL_DIR)
        if arguments.evaluate_only
        else MODEL_ID
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=True,
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_token_type_ids=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    print("\nПОДГОТОВКА ДАННЫХ")
    print("Задача: бинарная классификация тональности")
    print("Датасет:", DATASET_ID)
    print("Базовая модель:", MODEL_ID)
    print("Классы:", ID2LABEL)

    for split_name in ["train", "validation", "test"]:
        labels = np.asarray(dataset[split_name]["label"])
        print(
            f"{split_name}: {len(labels)}, "
            f"negative={int(np.sum(labels == 0))}, "
            f"positive={int(np.sum(labels == 1))}"
        )

    print("Максимальная длина:", MAX_LENGTH)
    print("Класс токенизатора:", type(tokenizer).__name__)

    example_ids = tokenized_dataset["train"][0]["input_ids"]
    print("Число токенов первого примера:", len(example_ids))
    print(
        "Первые 20 токенов:",
        tokenizer.convert_ids_to_tokens(example_ids[:20]),
    )
    print("Первые 20 input_ids:", example_ids[:20])
    print(
        "Первые 20 значений attention_mask:",
        tokenized_dataset["train"][0]["attention_mask"][:20],
    )

    if arguments.evaluate_only:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            use_safetensors=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            local_files_only=True,
            use_safetensors=True,
        )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    training_arguments = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=False,
        report_to="none",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nПАРАМЕТРЫ ОБУЧЕНИЯ")
    print("Устройство:", trainer.args.device)
    print("Число эпох:", EPOCHS)
    print("Размер батча:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Оптимизатор: AdamW")
    print(
        "Обучаемых параметров:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )

    if not arguments.evaluate_only:
        train_result = trainer.train(
            resume_from_checkpoint=(
                True if arguments.resume else None
            )
        )

        trainer.save_model(str(MODEL_DIR))
        tokenizer.save_pretrained(str(MODEL_DIR))
        trainer.save_state()
        trainer.save_metrics("train", train_result.metrics)

        epoch_metrics = [
            row
            for row in trainer.state.log_history
            if "eval_accuracy" in row and "eval_f1" in row
        ]

        (OUTPUT_DIR / "epoch_metrics.json").write_text(
            json.dumps(
                epoch_metrics,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("\nВАЛИДАЦИЯ ПО ЭПОХАМ")
        for row in epoch_metrics:
            print(
                f"epoch={row.get('epoch')}, "
                f"accuracy={row['eval_accuracy']:.6f}, "
                f"f1={row['eval_f1']:.6f}"
            )

        print(
            "Выбранный checkpoint:",
            trainer.state.best_model_checkpoint,
        )
        print("Лучшая модель сохранена:", MODEL_DIR)

    validation_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["validation"],
        metric_key_prefix="validation",
    )

    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    print_metrics(
        "ВАЛИДАЦИОННАЯ ВЫБОРКА",
        validation_metrics,
        "validation",
    )
    print_metrics(
        "НЕЗАВИСИМАЯ ТЕСТОВАЯ ВЫБОРКА",
        test_metrics,
        "test",
    )

    metrics_report = {
        "dataset": DATASET_ID,
        "base_model": MODEL_ID,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            metrics_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    trained_model = trainer.model
    trained_model.eval()
    device = trainer.args.device

    custom_predictions = []

    print("\nИНФЕРЕНС НА СОБСТВЕННЫХ ТЕКСТАХ")

    for index, text in enumerate(CUSTOM_TEXTS, start=1):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            return_token_type_ids=False,
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            logits = trained_model(**encoded).logits
            probabilities = torch.softmax(
                logits,
                dim=-1,
            )[0].cpu().tolist()

        predicted_id = int(np.argmax(probabilities))
        predicted_label = ID2LABEL[predicted_id]

        print(f"\n{index}. {text}")
        print(f"P(NEGATIVE): {probabilities[0]:.6f}")
        print(f"P(POSITIVE): {probabilities[1]:.6f}")
        print("Предсказанная тональность:", predicted_label)

        custom_predictions.append(
            {
                "text": text,
                "predicted_label": predicted_label,
                "negative_probability": probabilities[0],
                "positive_probability": probabilities[1],
            }
        )

    (OUTPUT_DIR / "custom_predictions.json").write_text(
        json.dumps(
            custom_predictions,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nРезультаты сохранены:", OUTPUT_DIR)
    print("Выполнение задания 10 завершено.")


if __name__ == "__main__":
    main()