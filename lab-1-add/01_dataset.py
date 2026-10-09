from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["cat", "dog", "horse"]


def main():
    train_data = CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
    )

    test_data = CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
    )

    class_ids = [
        train_data.class_to_idx[name]
        for name in CLASS_NAMES
    ]

    train_indices = [
        i
        for i, label in enumerate(train_data.targets)
        if label in class_ids
    ]

    test_indices = [
        i
        for i, label in enumerate(test_data.targets)
        if label in class_ids
    ]

    train_counts = Counter(
        train_data.targets[i] for i in train_indices
    )

    test_counts = Counter(
        test_data.targets[i] for i in test_indices
    )

    print("\nДатасет: CIFAR-10")
    print("Выбранные классы:", ", ".join(CLASS_NAMES))
    print("Размер изображений: 32 x 32, RGB")

    print("\nКоличество изображений:")
    for name, class_id in zip(CLASS_NAMES, class_ids):
        print(
            f"{name}: "
            f"train={train_counts[class_id]}, "
            f"test={test_counts[class_id]}"
        )

    print(f"\nВсего train: {len(train_indices)}")
    print(f"Всего test: {len(test_indices)}")

    label_map = {
        old_label: new_label
        for new_label, old_label in enumerate(class_ids)
    }

    print("\nСоответствие меток для будущего обучения:")
    for name, old_label in zip(CLASS_NAMES, class_ids):
        print(f"{name}: {old_label} -> {label_map[old_label]}")

    fig, axes = plt.subplots(3, 5, figsize=(10, 6))

    for row, (name, class_id) in enumerate(
        zip(CLASS_NAMES, class_ids)
    ):
        example_indices = [
            i
            for i in train_indices
            if train_data.targets[i] == class_id
        ][:5]

        for col, index in enumerate(example_indices):
            image, _ = train_data[index]
            axes[row, col].imshow(image)
            axes[row, col].set_title(name)
            axes[row, col].axis("off")

    fig.suptitle("CIFAR-10: cat, dog, horse")
    fig.tight_layout()

    output_path = SCREENSHOTS_DIR / "1_dataset_examples.png"
    fig.savefig(output_path, dpi=150)
    print(f"\nПримеры изображений сохранены: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()