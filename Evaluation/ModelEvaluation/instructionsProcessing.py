import json
from getData import getData

data=getData('Data/machine_generated_instructions.json')
qc=[]
for item in data:
    qc.append({"instruction":item['Content'],"input":item['Question']})

filename = 'Data/EveluateData.json'

with open(filename, 'w') as f:
    # 使用 json.dump 方法将数据写入文件
    # indent 参数是可选的，用于美化输出，使 JSON 文件易于阅读
    json.dump(qc, f, indent=4)