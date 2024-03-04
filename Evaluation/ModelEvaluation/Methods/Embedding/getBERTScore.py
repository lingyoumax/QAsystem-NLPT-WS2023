import sys
import os

current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)

if current_dir_path not in sys.path:
    sys.path.insert(0, current_dir_path)
    
from evaluate import load

bertscore = load("bertscore")

def getBERTScore(references, candidates):
    results = bertscore.compute(predictions=candidates, references=references, lang="en")
    return sum(results['precision'])/len(results['precision']), sum(results['recall'])/len(results['recall']), sum(results['f1'])/len(results['f1'])