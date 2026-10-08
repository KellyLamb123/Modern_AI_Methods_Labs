from pathlib import Path
from torchvision.datasets import CIFAR10

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "cifar10_data"

DATA_DIR.mkdir(exist_ok=True)

train_dataset = CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=True
)

test_dataset = CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=True
)

print("Обучающих изображений:", len(train_dataset))
print("Тестовых изображений:", len(test_dataset))

print("\nДоступные классы:")

for index, class_name in enumerate(train_dataset.classes):
    print(f"{index}: {class_name}")

import numpy as np

selected_classes = {
    "cat": 3,
    "dog": 5,
    "horse": 7
}

train_labels = np.array(train_dataset.targets)
test_labels = np.array(test_dataset.targets)

for name, class_id in selected_classes.items():
    print(f"{name}: {class_id}")

total_images = 0

for name, class_id in selected_classes.items():

    train_count = np.sum(train_labels == class_id)
    test_count = np.sum(test_labels == class_id)

    total_images += train_count + test_count

    print(
        f"{name}: "
        f"train={train_count}, "
        f"test={test_count}"
    )

print(f"\nВсего изображений трёх классов: {total_images}")