from pathlib import Path
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt

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

SCREENSHOTS_DIR = BASE_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))

rng = np.random.default_rng(42)

for row, (class_name, class_id) in enumerate(selected_classes.items()):
    indices = np.where(train_labels == class_id)[0]

    selected_indices = rng.choice(
        indices,
        size=4,
        replace=False
    )

    for col, index in enumerate(selected_indices):

        image = train_dataset.data[index]

        axes[row, col].imshow(image)
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

fig.suptitle(
    "CIFAR-10: Cat, Dog, Horse",
    fontsize=16
)
plt.tight_layout()

output_path = SCREENSHOTS_DIR / "1.4_dataset_examples.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

print("Количество изображений: 12")
print("Количество классов: 3")
print(f"Изображение сохранено: {output_path}")

plt.show()

RANDOM_SEED = 42

TRAIN_PER_CLASS = 1800
VAL_PER_CLASS = 400

split_rng = np.random.default_rng(RANDOM_SEED)

train_indices = []
val_indices = []
test_indices = []

for class_name, class_id in selected_classes.items():

    indices = np.where(train_labels == class_id)[0]

    indices = split_rng.permutation(indices)

    train_indices.extend(
        indices[:TRAIN_PER_CLASS]
    )

    val_indices.extend(
        indices[
            TRAIN_PER_CLASS:
            TRAIN_PER_CLASS + VAL_PER_CLASS
        ]
    )

    class_test_indices = np.where(
        test_labels == class_id
    )[0]

    test_indices.extend(class_test_indices)

train_indices = np.array(train_indices)
val_indices = np.array(val_indices)
test_indices = np.array(test_indices)

train_indices = split_rng.permutation(train_indices)
val_indices = split_rng.permutation(val_indices)
test_indices = split_rng.permutation(test_indices)

print("Train:", len(train_indices))
print("Validation:", len(val_indices))
print("Test:", len(test_indices))

for class_name, class_id in selected_classes.items():

    train_count = np.sum(
        train_labels[train_indices] == class_id
    )

    val_count = np.sum(
        train_labels[val_indices] == class_id
    )

    test_count = np.sum(
        test_labels[test_indices] == class_id
    )

    print(
        f"{class_name}: "
        f"Train={train_count}, "
        f"Validation={val_count}, "
        f"Test={test_count}"
    )

    assert train_count == TRAIN_PER_CLASS
    assert val_count == VAL_PER_CLASS
    assert test_count == 1000

intersection = np.intersect1d(
    train_indices,
    val_indices
)

assert len(intersection) == 0

SPLIT_PATH = BASE_DIR / "cifar10_split_indices.npz"

np.savez_compressed(
    SPLIT_PATH,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices,
    class_ids=np.array(list(selected_classes.values())),
    random_seed=RANDOM_SEED
)

print("\n=== Сохранение результатов ===")
print(f"Индексы сохранены: {SPLIT_PATH}")