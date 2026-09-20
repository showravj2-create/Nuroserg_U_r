import torch
from torch import nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self,smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self,logits,target):
        p=torch.sigmoid(logits).flatten(1)

        y=target.float().flatten(1)

        intersection = (p*y).sum(1)
        denominator = p.sum(1)+y.sum(1)

        dice = (
            2*intersection + self.smooth
        )/(
            denominator + self.smooth
        )

        return (1- dice).mean()

def boundary_loss(logits,target):
    p = torch.sigmoid(logits)
    gx = F.pad(torch.abs(p[...,1:] - p[..., :-1]),
               (0,1,0,0)
    )
    gy = F.pad(
        torch.abs(p[..., 1:, :] - p[..., :-1, :]),
        (0,0,0,1)
    )

    tx=F.pad(
        torch.abs(target[..., 1:] - target[..., :-1]),
        (0,1,0,0)
    )

    ty = F.pad(
        torch.abs(target[..., 1:, :] - target[..., :-1, :]),
        (0,0,0,1)
    )

    return F.l1_loss(
        gx+gy,
        tx+ty
    )

class CompositeSegmentationLoss(nn.Module):

    def __init__(
            self,
            dice_weight=1.0,
            bce_weight=0.5,
            boundary_weight=0.2
    ):
        super().__init__()

        self.dw = dice_weight
        self.bw = bce_weight
        self.boundary_weight = boundary_weight
        self.dice=SoftDiceLoss()

    def forward(self,logits,target):
        dice = self.dice(logits,target)
        bce = F.binary_cross_entropy_with_logits(logits,target.float())
        boundary = boundary_loss(
            logits,target
        )

        return (
            self.dw * dice
            +
            self.bw * bce
            +
            self.boundary_weight * boundary
        )

