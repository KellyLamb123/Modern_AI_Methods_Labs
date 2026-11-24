from pathlib import Path

from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer, sent_tokenize


input_file = Path(__file__).with_name("lab02_data.txt")
text = input_file.read_text(encoding="utf-8")

sentences = sent_tokenize(text, language="english")

tokenizer = RegexpTokenizer(r"[A-Za-z]+")
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

tokens = []

for sentence in sentences:
    sentence_words = tokenizer.tokenize(sentence.lower())

    sentence_tokens = [
        stemmer.stem(word)
        for word in sentence_words
        if word not in stop_words
    ]

    if sentence_tokens:
        tokens.append(sentence_tokens)

dict_tokens = corpora.Dictionary(tokens)

doc_term_mat = [
    dict_tokens.doc2bow(token)
    for token in tokens
]

num_topics = 2

ldamodel = models.ldamodel.LdaModel(
    corpus=doc_term_mat,
    num_topics=num_topics,
    id2word=dict_tokens,
    passes=25,
    random_state=42,
)

print("Исходный файл:", input_file.name)
print("Количество предложений:", len(sentences))
print("Документов после обработки:", len(tokens))
print("Размер словаря:", len(dict_tokens))
print("Количество тем:", num_topics)
print("Количество проходов:", 25)

print("\nПо пять представительных слов каждой темы:")

topics = ldamodel.show_topics(
    num_topics=num_topics,
    num_words=5,
    formatted=False,
)

for topic_id, topic_words in topics:
    print(f"\nТема {topic_id + 1}:")
    for word, weight in topic_words:
        print(f"{word:<20} {weight:.6f}")

assert len(topics) == 2
assert all(len(topic_words) == 5 for _, topic_words in topics)

print("\nПроверка количества тем и слов: OK")