from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer


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

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print("Исходный набор слов:")
print(", ".join(input_words))

print(
    f"\n{'Слово':<15}"
    f"{'Porter':<15}"
    f"{'Lancaster':<15}"
    f"{'Snowball':<15}"
)
print("-" * 60)

for word in input_words:
    print(
        f"{word:<15}"
        f"{porter.stem(word):<15}"
        f"{lancaster.stem(word):<15}"
        f"{snowball.stem(word):<15}"
    )