import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR / "feature_extraction_metrics.json").open(
    encoding="utf-8"
) as file:
    feature_extraction = json.load(file)

with (BASE_DIR / "fine_tuning_metrics.json").open(
    encoding="utf-8"
) as file:
    fine_tuning = json.load(file)

metrics = [
    ("Accuracy", "accuracy"),
    ("Precision (macro)", "precision_macro"),
    ("Recall (macro)", "recall_macro"),
]

print("Сравнение на одной тестовой выборке: 3000 изображений")
print("Значения метрик представлены в процентах.\n")

print(
    f"{'Метрика':<20}"
    f"{'Feature Extraction':>20}"
    f"{'Fine-tuning':>15}"
    f"{'Разница, п.п.':>16}"
)
print("-" * 71)

for name, key in metrics:
    before = feature_extraction[key] * 100
    after = fine_tuning[key] * 100

    print(
        f"{name:<20}"
        f"{before:>20.2f}"
        f"{after:>15.2f}"
        f"{after - before:>+16.2f}"
    )