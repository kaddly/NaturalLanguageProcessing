import torch
from torch import nn
from torch.nn import functional as F
from GPT import sequence_mask
import time
from datetime import timedelta


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    # `pred` 的形状：(`batch_size`, `num_steps`, `vocab_size`)
    # `label` 的形状：(`batch_size`, `num_steps`)
    # `valid_len` 的形状：(`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def grad_clipping(net, theta):  # @save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = y_hat.to(dtype=y.dtype) == y
    cmp = cmp.to(dtype=y.dtype)
    return float(cmp.sum()) / len(cmp)


def evaluate_accuracy_gpu(net, data_iter, token_loss, fineTurn, theta, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions

    with torch.no_grad():
        acc, loss = [], []
        for batch in data_iter:
            tokens, segments, valid_lens, labels = [x.to(device) for x in batch]
            y_hat, y_labels = net(tokens[:, :-1], segments[:, :-1])
            if fineTurn:
                token_l = token_loss(y_hat, tokens[:, 1:], valid_lens)
                nsp_l = F.cross_entropy(y_labels, labels)
                l = token_l + theta * nsp_l
            else:
                l = token_loss(y_hat, tokens[:, 1:], valid_lens)
            acc.append(accuracy(y_labels, labels))
            loss.append(l.sum() / valid_lens.sum())
    return sum(acc) / len(acc), sum(loss) / len(loss)


def train_GPT(net, train_iter, test_iter, num_epochs, fineTurn, lr, devices, theta=0.2):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    token_loss = MaskedSoftmaxCELoss()
    nsp_loss = nn.CrossEntropyLoss()
    start_time = time.time()
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            tokens, segments, valid_lens, labels = [x.to(devices[0]) for x in batch]
            y_hat, y_labels = net(tokens[:, :-1], segments[:, :-1])
            if fineTurn:
                token_l = token_loss(y_hat, tokens[:, 1:], valid_lens)
                nsp_l = nsp_loss(y_labels, labels)
                loss = token_l + theta * nsp_l
            else:
                loss = token_loss(y_hat, tokens[:, 1:], valid_lens)
            loss.sum().backward()
            grad_clipping(net, 1)
            optimizer.step()
            if total_batch % 50 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                train_acc = accuracy(y_labels, labels)
                dev_acc, dev_loss = evaluate_accuracy_gpu(net, test_iter, token_loss, fineTurn, theta)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(net.state_dict(), './saved_dict/BiRNN.ckpt')
                    improve = '*'
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, l.sum() / valid_lens.sum(), train_acc, dev_loss, dev_acc, time_dif,
                                 improve))

                net.train()
            total_batch += 1


def predict_GPT():
    pass
