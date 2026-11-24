import random

from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names


SEED = 42
TRAIN_RATIO = 0.8


def extract_features(name, n):
    return {"suffix": name.lower()[-n:]}


male_list = [
    (name, "male")
    for name in names.words("male.txt")
]
female_list = [
    (name, "female")
    for name in names.words("female.txt")
]

data = male_list + female_list
random.Random(SEED).shuffle(data)

split_index = int(TRAIN_RATIO * len(data))
train_names = data[:split_index]
test_names = data[split_index:]

input_names = ["Amelia", "Mark", "Sofia", "Robert"]

results = []

for n in range(1, 6):
    train_features = [
        (extract_features(name, n), label)
        for name, label in train_names
    ]
    test_features = [
        (extract_features(name, n), label)
        for name, label in test_names
    ]

    classifier = NaiveBayesClassifier.train(train_features)
    accuracy = nltk_accuracy(classifier, test_features)

    predictions = [
        classifier.classify(extract_features(name, n))
        for name in input_names
    ]

    results.append((n, accuracy, predictions))

print("Seed:", SEED)
print("Обучающих примеров:", len(train_names))
print("Тестовых примеров:", len(test_names))
print("Имена для финального тестирования:", ", ".join(input_names))

print(f"\n{'N':<4}{'Accuracy':<12}", end="")
for name in input_names:
    print(f"{name:<12}", end="")
print()
print("-" * 64)

for n, accuracy, predictions in results:
    print(f"{n:<4}{accuracy:<12.4f}", end="")
    for prediction in predictions:
        print(f"{prediction:<12}", end="")
    print()

best_result = max(
    results,
    key=lambda result: (result[1], -result[0]),
)

print("\nЛучшее N:", best_result[0])
print(f"Наибольшая тестовая точность: {best_result[1]:.4f}")
print("При равенстве точности выбирается меньшее N.")

assert len(results) == 5
assert all(0.0 <= result[1] <= 1.0 for result in results)
print("Проверка экспериментов: OK")