from nltk.stem import WordNetLemmatizer


input_words = [
    "calves",
    "wolves",
    "leaves",
    "running",
    "singing",
    "studies",
    "studying",
    "happily",
    "relational",
    "better",
    "went",
    "mice",
]

lemmatizer = WordNetLemmatizer()

print("Исходный набор слов:")
print(", ".join(input_words))

print(
    f"\n{'Слово':<15}"
    f"{'Существительное':<20}"
    f"{'Глагол':<20}"
)
print("-" * 55)

for word in input_words:
    noun = lemmatizer.lemmatize(word, pos="n")
    verb = lemmatizer.lemmatize(word, pos="v")
    print(f"{word:<15}{noun:<20}{verb:<20}")