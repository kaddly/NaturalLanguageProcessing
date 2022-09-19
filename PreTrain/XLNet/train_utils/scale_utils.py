import torch
import math
import collections


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


def bleu(pred_seq, label_seq, k):  # @save
    """计算 BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
