from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
'''
    not finished yet
'''

tokenizer = AutoTokenizer.from_pretrained('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')
model = AutoModelForSequenceClassification.from_pretrained('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')


premise = "What are some challenges and limitations in using data mining techniques in healthcare?"
hypothesis = "Some challenges and limitations in using data mining techniques in healthcare include the reliability of medical data, data sharing issues between healthcare organizations, and inappropriate modeling that can lead to inaccurate predictions."


inputs = tokenizer(premise, hypothesis, return_tensors='pt', padding=True, truncation=True, max_length=256)


with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits


probabilities = torch.nn.functional.softmax(logits, dim=1)


entailment_prob = probabilities[:, 0].item()
print(f"Entailment Probability: {entailment_prob}")
print(f"Hypothesis entails the premise: {bool(entailment_prob >= 0.5)}")
