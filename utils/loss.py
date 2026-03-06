# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:56:35 2024

@author: gbarbosa
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from scipy.ndimage.morphology import distance_transform_edt as edt
torch.set_default_dtype(torch.float32)

class DiceLoss(nn.Module):
   def __init__(self, weight=None, size_average=True):
       super(DiceLoss, self).__init__()

   def forward(self, inputs, targets, smooth=1):

       inputs = F.sigmoid(inputs)

       inputs = inputs.view(-1)
       targets = targets.view(-1)

       intersection = (inputs * targets).sum()
       dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
       return 1 - dice

class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JaccardLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        jac = (intersection + smooth)/(inputs.sum() + targets.sum() - intersection + smooth)
        return 1 - jac

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)
        with torch.no_grad():
            pred_dt = torch.from_numpy(self.distance_field(pred.cpu().numpy())).float()
            target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance.to('cuda')
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss