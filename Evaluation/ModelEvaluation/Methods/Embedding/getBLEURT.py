import sys
import os

current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)

if current_dir_path not in sys.path:
    sys.path.insert(0, current_dir_path)

from evaluate import load

bleurt = load("bleurt")

def getBLEURT(references, candidates):
    
    results = bleurt.compute(predictions=candidates, references=references)
    return results['scores']