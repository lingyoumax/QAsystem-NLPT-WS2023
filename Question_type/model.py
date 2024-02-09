from transformers import BertForSequenceClassification
import torch
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BERTMultiLabelBinaryClassification(torch.nn.Module):
    def __init__(self, num_labels,label_weight):
        super(BERTMultiLabelBinaryClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.label_weight=label_weight
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if labels is not None:
            pos_weight = torch.tensor(self.label_weight)  # 根据需要调整权重
            pos_weight.to(device)
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_fct.to(device)
            loss = loss_fct(logits.view(-1, self.bert.config.num_labels), labels.view(-1, self.bert.config.num_labels))
            return loss
        else:
            return logits

# class BERTMultiLabelBinaryClassification(torch.nn.Module):
#     def __init__(self, num_labels):
#         super(BERTMultiLabelBinaryClassification, self).__init__()
#         self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         logits = outputs.logits
#         probas = self.sigmoid(logits)
#         return probas

# model = BERTMultiLabelBinaryClassification(num_labels=len(labels[0]))
