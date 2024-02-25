import numpy as np
def getWER(references, candidates):
    assert len(references) == len(candidates), (
        "The number of candidate sentences must match the number of "
        "reference sentences.")
    wer=[]
    for t in range(len(references)):
        ref_words = references[t].split()
        gen_words = candidates[t].split()
    
        d = np.zeros((len(ref_words) + 1, len(gen_words) + 1), dtype=int)
    
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(gen_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(gen_words) + 1):
                if ref_words[i - 1] == gen_words[j - 1]:
                    cost = 0
                else:
                    cost = 1
                d[i][j] = min(
                    d[i - 1][j] + 1,     # deletion
                    d[i][j - 1] + 1,     # insertion
                    d[i - 1][j - 1] + cost  # substitution
                )

        # The WER is the edit distance divided by the number of words in the reference
        wer .append(d[len(ref_words)][len(gen_words)] / float(len(ref_words)))
    return wer