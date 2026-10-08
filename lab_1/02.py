import numpy as np
from sklearn import preprocessing

# Предоставление меток входных данных
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

# Создание кодировщика и установление соответствия между метками и числами
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Вывод отображения
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

# Преобразование меток с помощью кодировщика
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

# Декодирование набора чисел с помощью декодера
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))