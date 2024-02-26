from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
'''
    这里是检查参考文本和答案的蕴含性，就是看答案是否由参考文本得出，不是瞎说的。
    这里还需要做一个基准测试，人工标注10-20个参考文本-答案对，得出一个threshold。大于说明蕴含效果好，小于说明蕴含效果不好，即答案不是完全按照参考文本得出
'''

tokenizer = AutoTokenizer.from_pretrained('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')
model = AutoModelForSequenceClassification.from_pretrained('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')


premise = "Some challenges and limitations include the reliability of medical data, limitations around predictive modeling, and data sharing between healthcare organizations."

hypothesis = ("Our results show that the limitations of data mining in healthcare include reliability of medical data, "
              "data sharing between healthcare organizations, inappropriate modelling leading to inaccurate predictions. "
              "We conclude that there are many pitfalls in the use of data mining in healthcare and more work is needed "
)


inputs = tokenizer(premise, hypothesis, return_tensors='pt', padding=True, truncation=True, max_length=256)


with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits


probabilities = torch.nn.functional.softmax(logits, dim=1)


entailment_prob = probabilities[:, 0].item()
print(f"Entailment Probability: {entailment_prob}")
