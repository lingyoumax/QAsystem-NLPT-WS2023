import math
from collections import Counter


def calculate_entropy(text):
    """
    calculate the information density in a sentence
    :param text: string
    :return: number
    """
    words = text.split()
    word_counts = Counter(words)
    total_words = len(words)

    entropy = 0
    for count in word_counts.values():
        probability = count / total_words
        entropy -= probability * math.log2(probability)

    return entropy



answer = "Some challenges and limitations in using data mining techniques in healthcare include the reliability of medical data, data sharing issues between healthcare organizations, and inappropriate modeling that can lead to inaccurate predictions."


entropy = calculate_entropy(answer)
print(f"Entropy: {entropy:.2f}")
