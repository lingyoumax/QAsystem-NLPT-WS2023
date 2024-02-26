from transformers import pipeline
import math
from collections import Counter

'''
    方法1: 将答案通过summary后压缩后，与原始答案的长度词书做对比，如果长度差不多则说明不冗余。
    方法2: 检查答案中的信息商，低信息商表明冗余高。
'''


def summary(answer):
    """
    summary a sentence and return: the length of summarized sentence / original sentence
    :param answer: str
    :return: number
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(answer, max_length=len(answer), min_length=2, do_sample=False)

    return len(summary[0]['summary_text'].split()) / len(answer.split())


def calculate_entropy(text):
    """
    calculate the information density in a sentence
    :param text: string
    :return: number, number
    """
    words = text.split()
    word_counts = Counter(words)
    total_words = len(words)

    entropy = 0
    for count in word_counts.values():
        probability = count / total_words
        entropy -= probability * math.log2(probability)

    return entropy, math.log2(total_words)


answer = "Some challenges and limitations in using data mining techniques in healthcare include the reliability of medical data, data sharing issues between healthcare organizations, and inappropriate modeling that can lead to inaccurate predictions."

entropy = calculate_entropy(answer)
# print(f"Entropy: {entropy:.2f}")
print(entropy[0] / entropy[1])