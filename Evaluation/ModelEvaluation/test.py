from getData import getData
from Methods.LanguageModel.getSafety import getSafety
from collections import defaultdict
from Methods.LanguageModel.getStability import getStability
from Methods.Embedding.getBERTScore import getBERTScore
from Methods.LanguageModel.getFaithfulness import getFaithfulness
from Methods.LanguageModel.getRedundancy import getRedundancy
from Methods.Word.getSpelling import getSpelling
from config import GLOBAL_CONFIG
from tqdm import tqdm
import json

models=['SFTV3Model']
'''
#Safety
score=getData('Data/eveluateResults_safety.json')
for modelname in tqdm(models):
    data=getData(f"Data/data_{modelname}.json")
    safety=0
    for qa in data:
        safety=safety+getSafety(qa['input'],qa['output'])
    safety=safety/len(data)
    score[modelname]=safety.item()

filename = 'Data/eveluateResults_safety.json'

with open(filename, 'w') as f:
    json.dump(score, f, indent=4)

#Correctness
num_rep=GLOBAL_CONFIG['num_rep']
num_ref=GLOBAL_CONFIG['num_ref']
score=getData('Data/eveluateResults_correctness.json')
references=getData("Data/eveluateData.json")
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

filename = 'Data/eveluateResults_correctness.json'
with open(filename, 'w') as f:
    json.dump(score, f, indent=4)
'''
#Faithfulness
score=getData('Data/eveluateResults_faithfulness.json')
for modelname in tqdm(models):
    data=getData(f"Data/data_{modelname}.json")
    faithfulness=0
    ras=[[[qa['instruction'],qa['output']]] for qa in data]

    faithfulness=getFaithfulness(ras)
    score[modelname]=faithfulness
filename = 'Data/eveluateResults_faithfulness.json'

# 打开文件用于写入，'w' 表示写入模式
with open(filename, 'w') as f:
    # 使用 json.dump 方法将数据写入文件
    # indent 参数是可选的，用于美化输出，使 JSON 文件易于阅读
    json.dump(score, f, indent=4)


#Spelling
score=getData('Data/eveluateResults_spelling.json')
for modelname in tqdm(models):
    data=getData(f"Data/data_{modelname}.json")
    spelling=0
    for qa in data:
        spelling=spelling+getSpelling(qa['output'])
    spelling=spelling/len(data)
    score[modelname]=spelling

filename = 'Data/eveluateResults_spelling.json'

# 打开文件用于写入，'w' 表示写入模式
with open(filename, 'w') as f:
    # 使用 json.dump 方法将数据写入文件
    # indent 参数是可选的，用于美化输出，使 JSON 文件易于阅读
    json.dump(score, f, indent=4)