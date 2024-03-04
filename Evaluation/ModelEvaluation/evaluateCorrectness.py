from getData import getData
from Methods.Embedding.getBERTScore import getBERTScore
from config import GLOBAL_CONFIG
from tqdm import tqdm
import json
models=GLOBAL_CONFIG['models']
num_rep=GLOBAL_CONFIG['num_rep']
num_ref=GLOBAL_CONFIG['num_ref']
score={}
references=getData("Data/evaluateData.json")
references=[references[i]['output'] for i in range(num_ref) for _ in range(num_rep)]
batch_size=5
for modelname in tqdm(models):
    candidates=getData(f"Data/data_{modelname}.json")
    candidates=[d['output'] for d in candidates]
    precision=0
    recall=0
    f1=0
    for i in range(0,num_ref*num_rep,batch_size):
        p,r,f=getBERTScore(references[i:i+batch_size],candidates[i:i+batch_size])
        precision=precision+p
        recall=recall+r
        f1=f1+f

    score[modelname]={'precision':precision*batch_size/(num_ref*num_rep),'recall':recall*batch_size/(num_ref*num_rep),'f1':f1*batch_size/(num_ref*num_rep)}

filename = 'Data/evaluateResults_correctness.json'
with open(filename, 'w') as f:
    json.dump(score, f, indent=4)