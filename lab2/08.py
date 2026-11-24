from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize


def extract_features(words):
    return dict((word, True) for word in words)


fileids_pos = movie_reviews.fileids("pos")
fileids_neg = movie_reviews.fileids("neg")

features_pos = [
    (
        extract_features(movie_reviews.words(fileids=[file_id])),
        "Positive",
    )
    for file_id in fileids_pos
]

features_neg = [
    (
        extract_features(movie_reviews.words(fileids=[file_id])),
        "Negative",
    )
    for file_id in fileids_neg
]

num_pos = int(0.8 * len(features_pos))
num_neg = int(0.8 * len(features_neg))

features_train = features_pos[:num_pos] + features_neg[:num_neg]
features_test = features_pos[num_pos:] + features_neg[num_neg:]

classifier = NaiveBayesClassifier.train(features_train)

accuracy = nltk_accuracy(classifier, features_test)

print("Обучающих положительных отзывов:", num_pos)
print("Обучающих отрицательных отзывов:", num_neg)
print("Всего обучающих отзывов:", len(features_train))
print("Всего тестовых отзывов:", len(features_test))
print(f"Accuracy: {accuracy:.4f}")

print("\n20 наиболее информативных признаков:")
classifier.show_most_informative_features(20)

input_reviews = [
    "The film was wonderful and the actors were excellent.",
    "The plot was boring and the acting was terrible.",
    "I loved the beautiful music and the touching story.",
    "The film was not good and I would never watch it again.",
]

print("\nПредсказания для собственных отзывов:")

for index, review in enumerate(input_reviews, start=1):
    tokens = word_tokenize(review.lower(), language="english")
    features = extract_features(tokens)

    distribution = classifier.prob_classify(features)
    predicted_label = distribution.max()

    print(f"\n{index}. {review}")
    print(f"P(Positive): {distribution.prob('Positive'):.6f}")
    print(f"P(Negative): {distribution.prob('Negative'):.6f}")
    print("Предсказанная тональность:", predicted_label)

    probability_sum = sum(
        distribution.prob(label)
        for label in distribution.samples()
    )
    assert abs(probability_sum - 1.0) < 1e-6

print("\nПроверка вероятностей: OK")