from evaluate import load

bertscore = load("bertscore")

def getBERTScore(references, candidates):
    results = bertscore.compute(predictions=candidates, references=references, lang="en")
    return results['precision'], results['recall'], results['f1']