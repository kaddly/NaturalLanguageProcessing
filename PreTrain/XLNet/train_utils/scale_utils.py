import torch
import d2l.torch


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = y_hat.to(dtype=y.dtype) == y
    cmp = cmp.to(dtype=y.dtype)
    return float(cmp.sum()) / len(cmp)


def true_positive(pred, target, num_class):
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    out = []
    for i in range(num_class):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out)


def true_negative(pred, target, num_class):
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    out = []
    for i in range(num_class):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out)


def false_positive(pred, target, num_class):
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    out = []
    for i in range(num_class):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out)


def false_negative(pred, target, num_class):
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    out = []
    for i in range(num_class):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out)


def precision(pred, target, num_classes):
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)

    tp = true_positive(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def recall(pred, target, num_classes):
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fn = false_negative(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out


def f_beta_score(pred, target, num_classes, beta=1):
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    score = (1+beta**2)/(beta**2/rec+1/prec)
    score[torch.isnan(score)] = 0

    return score
