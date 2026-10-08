import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "cifar10_data"
SPLIT_PATH = BASE_DIR / "cifar10_split_indices.npz"

SCREENSHOTS_DIR = BASE_DIR / "screenshots"
OUTPUT_DIR = BASE_DIR / "transfer_learning_outputs"
RESULTS_DIR = BASE_DIR / "results"

for directory in [SCREENSHOTS_DIR, OUTPUT_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "feature_extraction_best.pth"

RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

CLASS_NAMES = ["cat", "dog", "horse"]

LABEL_MAP = {
    3: 0,
    5: 1,
    7: 2
}

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("\n=== Параметры эксперимента ===")
print("Device:", device)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)

weights = MobileNet_V3_Small_Weights.DEFAULT

transform = weights.transforms()

train_source = CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=transform
)

test_source = CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=transform
)

split_data = np.load(SPLIT_PATH)

train_indices = split_data["train_indices"]
val_indices = split_data["val_indices"]
test_indices = split_data["test_indices"]

class SelectedCIFAR10(Dataset):

    def __init__(self, source_dataset, indices):
        self.source_dataset = source_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        original_index = int(self.indices[index])

        image, original_label = self.source_dataset[
            original_index
        ]

        new_label = LABEL_MAP[original_label]

        return image, new_label


train_data = SelectedCIFAR10(
    train_source,
    train_indices
)

val_data = SelectedCIFAR10(
    train_source,
    val_indices
)

test_data = SelectedCIFAR10(
    test_source,
    test_indices
)


train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\n=== Размеры выборок ===")
print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))


# ==========================================
# 3.2.5. Создание нейросети
# ==========================================

model = mobilenet_v3_small(weights=weights)

# Замораживаем ВСЕ предобученные параметры
for parameter in model.parameters():
    parameter.requires_grad = False

# Определяем количество входных признаков
in_features = model.classifier[-1].in_features

# Заменяем последний слой на новый
model.classifier[-1] = nn.Linear(
    in_features,
    len(CLASS_NAMES)
)

# Переносим модель на GPU
model = model.to(device)

# Подсчитываем параметры
total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n=== Feature Extraction ===")
print("Всего параметров:", total_params)
print("Обучаемых параметров:", trainable_params)
print("Выходных классов:", len(CLASS_NAMES))

# Проверяем, что базовая сеть заморожена
assert all(
    not p.requires_grad
    for p in model.features.parameters()
)

print("Веса базовой сети заморожены!")


# ==========================================
# 3.2.6. Функция потерь и оптимизатор
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier[-1].parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 3.3. Функция одной эпохи
# ==========================================

def run_epoch(loader, training=True):

    if training:
        model.train()

        # Не обновляем статистику BatchNorm
        # в предобученной базовой сети
        model.features.eval()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        if training:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

        else:

            with torch.inference_mode():
                outputs = model(images)
                loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ==========================================
# 3.3. Обучение модели
# ==========================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_accuracy = -1.0
best_epoch = 0

print("\n=== Начало обучения ===")

for epoch in range(EPOCHS):

    # Обучение
    train_loss, train_acc = run_epoch(
        train_loader,
        training=True
    )

    # Валидация
    val_loss, val_acc = run_epoch(
        val_loader,
        training=False
    )

    # Сохраняем историю
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_acc:.4f}",
        flush=True
    )

    # Сохраняем лучшую модель
    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Лучшая модель сохранена!", flush=True)


print("\n=== Обучение завершено ===")
print("Лучшая эпоха:", best_epoch)
print(f"Лучшая Validation Accuracy: {best_val_accuracy:.4f}")


# ==========================================
# 3.4. Загрузка лучшей модели
# ==========================================

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


# ==========================================
# 3.4.1. Тестирование модели
# ==========================================

y_true = []
y_pred = []

print("\n=== Тестирование модели ===")

with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        y_true.extend(labels.tolist())

        y_pred.extend(
            predictions.cpu().tolist()
        )


# ==========================================
# 3.4.2. Расчёт метрик
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n=== Результаты Feature Extraction ===")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("\n=== Classification Report ===")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ==========================================
# 3.4.3. Матрица ошибок
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2]
)

print("\n=== Confusion Matrix ===")
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("Feature Extraction - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    SCREENSHOTS_DIR / "3.4_confusion_matrix.png",
    dpi=200
)

plt.close(fig)


# ==========================================
# 3.5. Графики обучения
# ==========================================

epochs_range = range(1, EPOCHS + 1)

# График точности
plt.figure(figsize=(9, 5))

plt.plot(
    epochs_range,
    train_accuracies,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Feature Extraction - Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    SCREENSHOTS_DIR / "3.3_training_accuracy.png",
    dpi=200
)

plt.close()


# График функции потерь
plt.figure(figsize=(9, 5))

plt.plot(
    epochs_range,
    train_losses,
    marker="o",
    label="Train Loss"
)

plt.plot(
    epochs_range,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Feature Extraction - Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    SCREENSHOTS_DIR / "3.3_training_loss.png",
    dpi=200
)

plt.close()


# ==========================================
# 3.5. Сохранение результатов
# ==========================================

results = {
    "architecture": "MobileNetV3 Small",
    "strategy": "Feature Extraction",
    "classes": CLASS_NAMES,
    "train_size": len(train_data),
    "validation_size": len(val_data),
    "test_size": len(test_data),
    "epochs": EPOCHS,
    "best_epoch": best_epoch,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "trainable_parameters": trainable_params,
    "best_validation_accuracy": best_val_accuracy,
    "accuracy": accuracy,
    "precision_macro": precision,
    "recall_macro": recall
}

results_path = (
    RESULTS_DIR / "feature_extraction_metrics.json"
)

with open(results_path, "w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\n=== Сохранение результатов ===")

print("Модель:", MODEL_PATH)
print("Метрики:", results_path)
print("Графики и матрица:", SCREENSHOTS_DIR)

print("\nFeature Extraction успешно завершён!")