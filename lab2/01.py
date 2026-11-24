from pprint import pprint

from nltk.tokenize import (
    WordPunctTokenizer,
    sent_tokenize,
    word_tokenize,
)


input_text = (
    "I can't stop learning Python. "
    "It's useful for text-processing, isn't it? "
    "Let's compare words, sentences, and punctuation!"
)

sentence_tokens = sent_tokenize(input_text, language="english")
word_tokens = word_tokenize(input_text, language="english")
word_punct_tokens = WordPunctTokenizer().tokenize(input_text)

print("Исходный текст:")
print(input_text)

print("\n1. sent_tokenize:")
pprint(sentence_tokens, width=100)
print("Количество предложений:", len(sentence_tokens))

print("\n2. word_tokenize:")
pprint(word_tokens, width=100)
print("Количество токенов:", len(word_tokens))

print("\n3. WordPunctTokenizer:")
pprint(word_punct_tokens, width=100)
print("Количество токенов:", len(word_punct_tokens))