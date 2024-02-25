from transformers import BertForSequenceClassification
import torch
from classification.loss import MixedLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERTMultiLabelBinaryClassification_FactorLoss(torch.nn.Module):
    def __init__(self, num_labels, label_weight, gamma=2.0, mix_ratio=0.5):
        super(BERTMultiLabelBinaryClassification_FactorLoss, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.label_weight = torch.tensor(label_weight)
        self.gamma = gamma
        self.mixed_loss = MixedLoss(weight=self.label_weight, gamma=self.gamma, pos_weight=self.label_weight,
                                    mix_ratio=mix_ratio).to(device)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if labels is not None:
            loss = self.mixed_loss(logits.view(-1, self.bert.config.num_labels),
                                   labels.view(-1, self.bert.config.num_labels))
            return loss
        else:
            return logits


class BERTMultiLabelBinaryClassification(torch.nn.Module):
    def __init__(self, num_labels, label_weight):
        super(BERTMultiLabelBinaryClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.label_weight = label_weight

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if labels is not None:
            pos_weight = torch.tensor(self.label_weight)
            pos_weight.to(device)
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_fct.to(device)
            loss = loss_fct(logits.view(-1, self.bert.config.num_labels), labels.view(-1, self.bert.config.num_labels))
            return loss
        else:
            return logits
