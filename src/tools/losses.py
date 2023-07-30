import torch
from torch import nn
from torch.nn.functional import one_hot


class DiceLoss(nn.Module):
    """
    Dice Loss as per:
      https://www.jeremyjordan.me/semantic-segmentation/
    """

    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        intersection = (y_pred * y_true).sum() + self.epsilon
        denom = y_pred.sum() + y_true.sum() + self.epsilon
        loss = 1 - (2 * intersection / denom)

        return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss as per:
        https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
    """

    def __init__(self, epsilon=1e-4, alpha=0.8, gamma=1.5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, alpha=-1, gamma=-1):
        alpha = self.alpha if alpha == -1 else alpha
        gamma = self.gamma if gamma == -1 else gamma
        y_pred = torch.sigmoid(y_pred)

        tp = (y_pred * y_true).sum()
        fn = ((1 - y_pred) * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()

        tversky = (tp + self.epsilon) / (
            tp + alpha * fn + (1 - alpha) * fp + self.epsilon
        )
        loss = torch.pow((1 - tversky), gamma)

        return loss
