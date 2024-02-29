from utils import jload
from torch.utils.data import Dataset
import torch

def load_dataset(path):
    dataset=jload(path)
    pairs=[]
    for sample in dataset:
        pair={}
        prompt=sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"]=prompt + "\n" + chosen_summary
        pair["rejected"]=prompt + "\n" + rejected_summary
        pairs.append(pair)
    print("actual create {} num samples".format(len(pairs)))
    return pairs

class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in pairs:
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])
        print("dataset have {} samples".format(len(self.chosen_input_ids)))

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        chosen_input_ids=[]
        chosen_attn_masks=[]
        rejected_input_ids=[]
        rejected_attn_masks=[]
        if isinstance(idx,list):
            print("get {} instances".format(len(idx)))
            for i in idx:
                chosen_input_ids.append(self.chosen_input_ids[i])
                chosen_attn_masks.append(self.chosen_attn_masks[i])
                rejected_input_ids.append(self.rejected_input_ids[i])
                rejected_attn_masks.append(self.rejected_attn_masks[i])
            return (
                chosen_input_ids,
                chosen_attn_masks,
                rejected_input_ids,
                rejected_attn_masks,
            )
        else:
            return (
                self.chosen_input_ids[idx],
                self.chosen_attn_masks[idx],
                self.rejected_input_ids[idx],
                self.rejected_attn_masks[idx],
            )

class DataCollatorReward:
    def __call__(self, data):
        temp=[f[0] for f in data]
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch