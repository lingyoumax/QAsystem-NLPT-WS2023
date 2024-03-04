from textblob import TextBlob
def getSpelling(text):
    blob = TextBlob(text)
    corrected_text = blob.correct()

    original_words = text.split()
    corrected_words = str(corrected_text).split()

    num_errors = sum([1 for original, corrected in zip(original_words, corrected_words) if original != corrected])
    error_rate = num_errors / len(original_words) if len(original_words) else 0
    return error_rate