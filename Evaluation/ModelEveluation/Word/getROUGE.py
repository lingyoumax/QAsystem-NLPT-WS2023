from collections import Counter

def getROUGE(references, candidates, n=1):
    assert len(references) == len(candidates), (
        "The number of candidate sentences must match the number of "
        "reference sentences.")
    def ngrams(text, n):
        words = text.split()
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    recalls=[]
    precisions=[]
    fmeasures=[]
    for t in range(len(references)):
        ref_ngrams = ngrams(references[t], n)
        gen_ngrams = ngrams(candidates[t], n)

        ref_ngram_counts = Counter(ref_ngrams)
        gen_ngram_counts = Counter(gen_ngrams)

        overlap = sum((ref_ngram_counts & gen_ngram_counts).values())

        recalls.append(overlap / max(len(ref_ngrams), 1))
        precisions.append(overlap / max(len(gen_ngrams), 1))
        fmeasures.append(2*precisions[t]*recalls[t]/(precisions[t]+recalls[t]))

    return recalls, precisions, fmeasures