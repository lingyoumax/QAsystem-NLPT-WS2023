from transformers import pipeline


def summary(answer):
    """
    summary a sentence and return: the length of summarized sentence / original sentence
    :param answer: str
    :return: number
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(answer, max_length=len(answer), min_length=2, do_sample=False)

    return len(summary[0]['summary_text'].split()) / len(answer.split())

