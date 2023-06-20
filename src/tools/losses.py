import torch
from torch import nn
from torch.nn.functional import one_hot


class DiceLoss(nn.Module):
    """
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
    https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
    """

    def __init__(self, epsilon=1e-4, alpha=0.8, gamma=1.5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)

        tp = (y_pred * y_true).sum()
        fn = ((1 - y_pred) * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()

        tversky = (tp + self.epsilon) / (
            tp + self.alpha * fn + (1 - self.alpha) * fp + self.epsilon
        )
        loss = torch.pow((1 - tversky), self.gamma)

        return loss