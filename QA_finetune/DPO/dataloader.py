from utils import jload
from torch.utils.data import Dataset
import torch

def load_dataset(path):
    dataset=jload(path)
    pairs={"prompt":[],"chosen":[],"rejected":[]}
    for sample in dataset:
        prompt=sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pairs["chosen"].append(chosen_summary)
        pairs["rejected"].append(rejected_summary)
        pairs["prompt"].append(prompt)
    print("actual create {} num samples".format(len(pairs)))
    return pairs
