from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
)
from sklearn.naive_bayes import MultinomialNB


categories = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
    "talk.politics.misc",
]

training_data = fetch_20newsgroups(
    subset="train",
    categories=categories,
    download_if_missing=False,
)

count_vectorizer = CountVectorizer()
train_counts = count_vectorizer.fit_transform(training_data.data)

tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)

input_sentences = [
    "The graphics software renders a three dimensional image.",
    "The baseball player hit a home run and won the game.",
    "The doctor discussed the symptoms and treatment of the disease.",
    "The spacecraft entered orbit around the planet.",
    "The government debated a new policy in parliament.",
]

classifier = MultinomialNB().fit(
    train_tfidf,
    training_data.target,
)

input_counts = count_vectorizer.transform(input_sentences)
input_tfidf = tfidf_transformer.transform(input_counts)

predictions = classifier.predict(input_tfidf)

print("Категории классификации:")
for index, name in enumerate(training_data.target_names):
    print(f"{index}: {name}")

print("\nКоличество обучающих документов:", len(training_data.data))
print("Размер обучающей матрицы:", train_tfidf.shape)
print("Размер матрицы новых предложений:", input_tfidf.shape)

print("\nРезультаты классификации:")
for index, (sentence, prediction) in enumerate(
    zip(input_sentences, predictions),
    start=1,
):
    category = training_data.target_names[prediction]
    print(f"\n{index}. {sentence}")
    print("Предсказанная категория:", category)

    if input_counts[index - 1].nnz == 0:
        print("Внимание: все слова предложения отсутствуют в словаре.")

assert train_tfidf.shape[1] == input_tfidf.shape[1]
print("\nПроверка согласованности признаков: OK")