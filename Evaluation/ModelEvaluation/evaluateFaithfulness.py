from getData import getData
from Methods.LanguageModel.getFaithfulness import getFaithfulness
from config import GLOBAL_CONFIG
from tqdm import tqdm
import json

models=GLOBAL_CONFIG['models']
score={}
for modelname in tqdm(models):
    data=getData(f"Data/data_{modelname}.json")
    faithfulness=0
    ras=[[[qa['instruction'],qa['output']]] for qa in data]

    faithfulness=getFaithfulness(ras)
    score[modelname]=faithfulness
filename = 'Data/evaluateResults_faithfulness.json'

# 打开文件用于写入，'w' 表示写入模式
with open(filename, 'w') as f:
    # 使用 json.dump 方法将数据写入文件
    # indent 参数是可选的，用于美化输出，使 JSON 文件易于阅读
    json.dump(score, f, indent=4)