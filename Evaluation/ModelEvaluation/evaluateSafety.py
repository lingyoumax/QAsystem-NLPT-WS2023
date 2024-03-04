from getData import getData
from Methods.LanguageModel.getSafety import getSafety
from config import GLOBAL_CONFIG
from tqdm import tqdm
import json

models=GLOBAL_CONFIG['models']
score={}
for modelname in tqdm(models):
    data=getData(f"Data/data_{modelname}.json")
    safety=0
    for qa in data:
        safety=safety+getSafety(qa['input'],qa['output'])
    safety=safety/len(data)
    score[modelname]=safety.item()

filename = 'Data/evaluateResults_safety.json'

with open(filename, 'w') as f:
    json.dump(score, f, indent=4)