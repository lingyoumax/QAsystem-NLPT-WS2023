from collections import Counter

def getBLEU(references, candidates, n=1):
    assert len(references) == len(candidates), (
        "The number of candidate sentences must match the number of "
        "reference sentences.")
    def ngrams(text, n):
        words = text.split()
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    precisions=[min(1,len(candidates[i].split())/len(references[i].split())) for i in range(len(references))]
    for t in range(len(references)):
        for i in range(1,n+1):
            ref_ngrams = ngrams(references[t], i)
            gen_ngrams = ngrams(candidates[t], i)

            ref_ngram_counts = Counter(ref_ngrams)
            gen_ngram_counts = Counter(gen_ngrams)

            overlap = sum((ref_ngram_counts & gen_ngram_counts).values())

            precisions[t]=precisions[t]*overlap / max(len(gen_ngrams), 1)

    return precisions