from transformers import BertForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MixedLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, pos_weight=None, mix_ratio=0.5):
        super(MixedLoss, self).__init__()
        self.focal_loss = FocalLoss(weight=weight, gamma=gamma,reduction="mean").to(device)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        self.mix_ratio = mix_ratio  

    def forward(self, inputs, targets):
        loss_focal = self.focal_loss(inputs, targets )
        loss_bce = self.bce_loss(inputs, targets)
        return (self.mix_ratio * loss_focal) + ((1 - self.mix_ratio) * loss_bce)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='none'):
        super(FocalLoss, self).__init__()
        self.weight = weight.to(device) 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss=BCE_loss.to(device)
        pt = torch.exp(-BCE_loss)  
        F_loss = self.weight * ((1 - pt) ** self.gamma) * BCE_loss

        if self.reduction == 'sum':
            return torch.sum(F_loss)
        elif self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss