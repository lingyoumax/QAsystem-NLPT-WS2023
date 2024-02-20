def getPrecision_Recall_Fmeasure(references,candidates):
    assert len(references) == len(candidates), (
        "The number of candidate sentences must match the number of "
        "reference sentences.")
    precisions=[]
    recalls=[]
    fmeasures=[]
    for t in range(len(references)):
        c = candidates[t].split()
        r = references[t].split()

        correct_words = len(set(c) & set(r))

        precisions.append(correct_words / len(c) if c else 0)
        recalls.append(correct_words / len(r) if r else 0)
        fmeasures.append((2 * precisions[t] * recalls[t]) / (precisions[t] + recalls[t]) if (precisions[t] + recalls[t]) else 0)

    return precisions, recalls, fmeasures