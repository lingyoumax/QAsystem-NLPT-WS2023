from SFTV0Model.inference import inference
import json
from getData import getData

def replicate_elements(lst, rep_num):
    return [x for x in lst for _ in range(rep_num)]

data=getData()

ref_num=100
rep_num=5
instructions=replicate_elements(data[:ref_num],rep_num)

results=inference(instructions)

filename = 'Data/data_SFTV0Model.json'

with open(filename, 'w') as f:
    json.dump(results, f, indent=4)