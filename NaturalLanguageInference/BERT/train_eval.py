import torch
from torch import nn
import time


def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = y_hat.to(dtype=y.dtype) == y
    cmp = cmp.to(dtype=y.dtype)
    return float(cmp.sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions

    with torch.no_grad():
        acc, nums = 0, 0
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            acc += accuracy(net(X), y)
            nums += len(y)
    return acc / nums


def train_batch_snli(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_snli(net, train_iter, test_iter, loss, trainer, num_epochs, devices):
    """Train a model with mutiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(init_weights)
    timer, num_batches = [], len(train_iter)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        ls, accs, batch_num, labels_num = 0, 0, 0, 0
        for i, (features, labels) in enumerate(train_iter):
            tik = time.time()
            l, acc = train_batch_snli(net, features, labels, loss, trainer, devices)
            ls += l
            accs += acc
            batch_num += labels.shape[0]
            labels_num += labels.numel()
            timer.append(time.time() - tik)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'epoch:{epoch + 1}>>'
                      f'step:{i + 1}>>'
                      f'loss:{ls / batch_num:.3f}>>'
                      f'train acc:{accs / labels_num:.3f}')
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch:{epoch + 1}>>test_acc:{test_acc:.3f}')
    print(f'loss {ls / batch_num:.3f}, train acc '
          f'{accs / labels_num:.3f}, test acc {test_acc:.3f}')
    print(f'{batch_num * num_epochs / sum(timer):.1f} examples/sec on '
          f'{str(devices)}')


def predict_snli(net, vocab, premise, hypothesis, device):
    """预测前提和假设之间的逻辑关系"""
    net.eval()
    premise = torch.tensor(vocab[premise], device=device)
    hypothesis = torch.tensor(vocab[hypothesis], device=device)
    label = torch.argmax(net([premise.reshape((1, -1)), hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 else 'neutral'
