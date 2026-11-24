from nltk.corpus import brown

from text_chunker import chunke


N = 1000
M = 100

words = list(brown.words()[:N])
chunks = chunke(words, M)

print("Количество исходных слов:", len(words))
print("Размер чанка в словах:", M)
print("Количество чанков:", len(chunks))
print("\nПервые 50 символов каждого чанка:")

for index, chunk in enumerate(chunks[:20]):
    print(
        f"Чанк {index:02d} "
        f"({len(chunk.split())} слов): "
        f"{chunk[:50]}"
    )

assert sum(len(chunk.split()) for chunk in chunks) == N
print("\nПроверка сохранения количества слов: OK")