import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):

    def __init__(self,alpha=0.25,gamma=2):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self,logits,targets):

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )

        prob = torch.sigmoid(logits)

        pt = targets*prob + (1-targets)*(1-prob)

        loss = self.alpha*(1-pt)**self.gamma*bce

        return loss.mean()


class MultiTaskLoss(nn.Module):

    def __init__(self,height_weight=0.3):

        super().__init__()

        self.height_weight = height_weight

        self.focal = FocalBCELoss()

    def dice_loss(self,pred,target):

        pred = torch.sigmoid(pred)

        smooth = 1e-5

        inter = (pred*target).sum()

        union = pred.sum()+target.sum()

        dice = (2*inter+smooth)/(union+smooth)

        return 1-dice

    def forward(self,occ_pred,occ_gt,h_pred,h_gt):

        focal = self.focal(occ_pred,occ_gt)

        dice = self.dice_loss(occ_pred,occ_gt)

        h_loss = F.mse_loss(h_pred,h_gt)

        total = focal + dice + self.height_weight*h_loss

        return total,focal,dice,h_loss