import sys
import os

current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)

if current_dir_path not in sys.path:
    sys.path.insert(0, current_dir_path) 

from moverscore.moverscore_v2 import get_idf_dict, word_mover_score
def calculate_moverscore(references, hypotheses, idf_dict):
    scores = word_mover_score(references, hypotheses, idf_dict, stop_words=[], n_gram=1, remove_subwords=True)
    return scores

# Example usage
references = ["This is a reference sentence for evaluation."]
hypotheses = ["This is a generated sentence to be evaluated."]
idf_dict = get_idf_dict(references)  # IDF values needed for weighted MoverScore

# Calculate MoverScore
scores = calculate_moverscore(references, hypotheses, idf_dict)
print(scores)
