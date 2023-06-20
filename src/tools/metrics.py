import numpy as np
import torch

def eval_metrics(pred, target, epsilon):
    target = torch.squeeze(torch.argmax(target, dim=1))
    pred = torch.squeeze(torch.argmax(pred, dim=1))

    tp = ((pred == 1) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = (2 * precision * recall) / (precision + recall + epsilon)
    
    intersection = (pred * target).sum() + epsilon
    union = pred.sum() + target.sum() - intersection + epsilon
    mIoU = intersection / union

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1_score,
        "mIoU": mIoU
    }