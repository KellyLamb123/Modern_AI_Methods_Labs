def chunke(words, chunk_size):
    """Разбивает последовательность слов на текстовые блоки."""
    if chunk_size <= 0:
        raise ValueError("Размер чанка должен быть положительным")

    words = list(words)

    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), chunk_size)
    ]