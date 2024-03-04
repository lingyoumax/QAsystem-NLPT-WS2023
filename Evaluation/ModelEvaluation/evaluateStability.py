from getData import getData
from collections import defaultdict
from Methods.LanguageModel.getStability import getStability
from config import GLOBAL_CONFIG
from tqdm import tqdm
import json

models=GLOBAL_CONFIG['models']
score={}

def similarity(values):
    if len(values) < 2:  
        return 1
    sim_values = 0
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            sim_values=sim_values+getStability(values[i],values[j])  
    return 2*sim_values/(len(values)*(len(values)-1))

for modelname in tqdm(models):
    data=getData(f"Data/data_{modelname}.json")
    groups = defaultdict(list)
    for item in data:
        groups[item['input']].append(item)

    stability=0
    for group, items in groups.items():
        output_values = [item['output'] for item in items]
        stability =stability + similarity(output_values)
    
    stability=stability/len(groups)
    score[modelname]=stability

filename = 'Data/evaluateResults_stability.json'

# 打开文件用于写入，'w' 表示写入模式
with open(filename, 'w') as f:
    # 使用 json.dump 方法将数据写入文件
    # indent 参数是可选的，用于美化输出，使 JSON 文件易于阅读
    json.dump(score, f, indent=4)