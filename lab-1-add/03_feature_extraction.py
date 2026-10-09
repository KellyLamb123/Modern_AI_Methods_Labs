import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["cat", "dog", "horse"]

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def main():
    torch.manual_seed(42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    screenshots_dir = BASE_DIR / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    train_data = CIFAR10(
        root=str(BASE_DIR / "data"),
        train=True,
        download=False,
        transform=transform,
    )

    test_data = CIFAR10(
        root=str(BASE_DIR / "data"),
        train=False,
        download=False,
        transform=transform,
    )

    class_ids = [
        train_data.class_to_idx[name]
        for name in CLASS_NAMES
    ]

    label_map = {
        old_label: new_label
        for new_label, old_label in enumerate(class_ids)
    }

    train_indices = [
        i for i, label in enumerate(train_data.targets)
        if label in class_ids
    ]

    test_indices = [
        i for i, label in enumerate(test_data.targets)
        if label in class_ids
    ]

    train_data.target_transform = label_map.__getitem__
    test_data.target_transform = label_map.__getitem__

    train_subset = Subset(train_data, train_indices)
    test_subset = Subset(test_data, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = mobilenet_v3_small(weights=weights)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(
        in_features,
        len(CLASS_NAMES),
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    print("Стратегия: Feature Extraction")
    print("Устройство:", device)
    print("Обучающих изображений:", len(train_subset))
    print("Тестовых изображений:", len(test_subset))
    print("Эпох:", EPOCHS)
    print("Размер пакета:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print(
        "Базовая сеть заморожена:",
        all(
            not parameter.requires_grad
            for parameter in model.features.parameters()
        ),
    )
    print("Выходной слой:", model.classifier[-1])

    for epoch in range(EPOCHS):
        model.train()

        model.features.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_number, (images, labels) in enumerate(
            train_loader, start=1
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

            if batch_number % 100 == 0:
                print(
                    f"Эпоха {epoch + 1}/{EPOCHS}: "
                    f"пакет {batch_number}/{len(train_loader)}",
                    flush=True,
                )

        print(
            f"Эпоха {epoch + 1}/{EPOCHS} завершена | "
            f"Loss: {total_loss / total:.4f} | "
            f"Train Accuracy: {correct / total:.4f}",
            flush=True,
        )

    model.eval()
    y_true = []
    y_pred = []

    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ),
    }

    print("\nРезультаты на тестовой выборке:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")

    model.cpu()
    torch.save(
        model.state_dict(),
        BASE_DIR / "feature_extraction.pth",
    )

    with (BASE_DIR / "feature_extraction_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    matrix = confusion_matrix(
        y_true, y_pred, labels=[0, 1, 2]
    )

    print("\nМатрица ошибок:")
    print(matrix)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Feature Extraction")
    plt.tight_layout()
    plt.savefig(
        screenshots_dir / "3_feature_extraction_matrix.png",
        dpi=150,
    )

    print("\nВеса, метрики и матрица ошибок сохранены.")
    plt.show()


if __name__ == "__main__":
    main()