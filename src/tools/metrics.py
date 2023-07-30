import torch


def eval_metrics(pred, target, epsilon=1e-4):
    """
    Evaluation of the prediction according to the target, evaluation metrics are:
    accuracy, precision, recall, F1, mIoU
    @param pred: the model prediction
    @param target: ground truth
    @param epsilon: epsilon parameter for accuracy, recall, precision and f1
    @return:
    """
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
    
    intersection = (pred * target).sum().item() + epsilon
    union = pred.sum().item() + target.sum().item() - intersection + epsilon
    mIoU = intersection / union

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1_score,
        "mIoU": mIoU
    }