
from transformers import pipeline

pipe=pipeline(model="manueldeprada/FactCC",device=0)
def getFaithfulness(data):
    output=pipe(data,truncation=True
                ,padding='max_length')
    scores=[item['score'] for item in output]
    return sum(scores)/len(scores)
    