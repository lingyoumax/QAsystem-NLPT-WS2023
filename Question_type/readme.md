# 1 Current Result  
- <img width="677" alt="7b93c244d9d958152ac8ae4f6433e2f" src="https://github.com/lingyoumax/QAsystem-NLPT-WS2023/assets/43053906/cda93b54-0fa6-4af3-80e5-f1278f85305b">  
# 2 Current progress
## 2.1 Models and methods used
- I.To address this problem, we use the Bert model for fine-tuning training.
- II.According to the characteristics of the data set (multi-category binary classification and category imbalance issues), focal loss and BCE loss weighting methods are used for training.
- III.In order to adapt to different data types, a step-by-step training method is adopted, using half single category and half multi-category, all multi-type, all single type and all data sets for training.
## 2.2 conclusion
Up to now, we believe that the effect of the model requires a better data set, especially the inspection found that the data set has some contamination problems, resulting in not very good results.
# 3 Directions for improvement
When generating the data set, we did not consider which combinations conformed to human questioning habits, resulting in a data set that was not very good. Therefore, the latter step can generate a more complete data set for training.
# 4 How to use
just load model model_state_dict_all_data_1.pth(https://drive.google.com/file/d/1oHAzWi0VMQVfd_TP8kNmCK4K72-KRcPi/view?usp=drive_link),and use tokenizer to deal the user's input. 
if single sentence:
```python
sentence_encodings = tokenizer(sentence, truncation=True, padding=True, max_length=128)
sentence_seq = torch.tensor(sentence_encodings['input_ids'])
sentence_mask = torch.tensor(sentence_encodings['attention_mask'])
model.eval()
with torch.no_grad():
    model.cpu()
    inputs = {
            'input_ids':sentence_seq.unsqueeze(0),
            'attention_mask':sentence_mask.unsqueeze(0)
        }
    outputs = model(**inputs)
    logits =  torch.sigmoid(outputs).detach().cpu().numpy()
pred_labels=(logits > 0.5).astype(int)
print(pred_labels)
```
if multiple sentence:
```python
sentence_encodings = tokenizer(sentence, truncation=True, padding=True, max_length=128)
sentence_seq = torch.tensor(sentence_encodings['input_ids'])
sentence_mask = torch.tensor(sentence_encodings['attention_mask'])
model.eval()
with torch.no_grad():
    model.cpu()
    inputs = {
            'input_ids':sentence_seq,
            'attention_mask':sentence_mask
        }
    outputs = model(**inputs)
    logits =  torch.sigmoid(outputs).detach().cpu().numpy()
pred_labels=(logits > 0.5).astype(int)
print(pred_labels)
```
