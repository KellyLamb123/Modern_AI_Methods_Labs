import csv
from pathlib import Path

import numpy as np
from nltk.corpus import brown
from sklearn.feature_extraction.text import CountVectorizer

from text_chunker import chunke


N = 1000
M = 100

words = list(brown.words()[:N])
chunks = chunke(words, M)

documents = [
    {"index": index, "text": chunk}
    for index, chunk in enumerate(chunks)
]

vectorizer = CountVectorizer()
document_term_matrix = vectorizer.fit_transform(
    [document["text"] for document in documents]
)

terms = vectorizer.get_feature_names_out()
matrix = document_term_matrix.toarray()

print("Количество документов:", len(documents))
print("Количество терминов:", len(terms))
print("Размер матрицы:", matrix.shape)

print("\nПолная терм-документная матрица:")
with np.printoptions(threshold=np.inf, linewidth=120):
    print(matrix)

output_file = Path(__file__).with_name("05_bow_matrix.csv")

with output_file.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["index", *terms])

    for document, row in zip(documents, matrix):
        writer.writerow([document["index"], *row.tolist()])

preview_terms = terms[:12]

print("\nФрагмент с названиями первых 12 терминов:")
print(f"{'index':>6}", end="")
for term in preview_terms:
    print(f"{term:>12}", end="")
print()

for document, row in zip(documents, matrix):
    print(f"{document['index']:>6}", end="")
    for value in row[:12]:
        print(f"{int(value):>12}", end="")
    print()

term_frequencies = matrix.sum(axis=0)
top_indices = np.argsort(-term_frequencies, kind="stable")[:10]

print("\n10 наиболее частых терминов:")
for index in top_indices:
    print(f"{terms[index]}: {int(term_frequencies[index])}")

assert matrix.shape[0] == 10
assert np.all(matrix >= 0)

print("\nПолная матрица сохранена:", output_file)
print("Проверка матрицы: OK")