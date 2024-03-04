from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def getRedundancy(answer):
    summary = summarizer(answer, max_length=len(answer), min_length=2, do_sample=False)

    return len(summary[0]['summary_text'].split()) / len(answer.split())
