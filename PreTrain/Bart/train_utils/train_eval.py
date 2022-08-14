import time
import os
import torch
from torch import nn
import torch.nn.functional as f
from datetime import timedelta
from .scale_utils import Accumulator, accuracy, MaskedSoftmaxCELoss
from .optimizer_utils import create_lr_scheduler, grad_clipping


def evaluate_accuracy_gpu(net, data_iter, vocab_size, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    with torch.no_grad():
        acc, loss = [], []
        for X, y in data_iter:
            if isinstance(X, tuple):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            _, y_hat = net(X[0], None, X[1])
            acc.append(accuracy(y_hat.reshape(-1, vocab_size), y.reshape(-1)))
            loss.append(f.cross_entropy(y_hat.reshape(-1, vocab_size), y.reshape(-1)))
    return sum(acc) / len(acc), sum(loss) / len(loss)


def train(net, train_iter, val_iter, lr, num_epochs, vocab, devices):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_iter), num_epochs)
    loss = MaskedSoftmaxCELoss()
    start_time = time.time()

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    metric = Accumulator(4)

    # 模型参数保存路径
    saved_dir = '../saved_dict'
    model_file = 'Bart'
    parameter_path = os.path.join(saved_dir, model_file)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(devices[0]) for x in batch]
            bos = torch.tensor([vocab['<bos>']] * Y.shape[0], device=devices[0]).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            train_loss = loss(Y_hat, Y, Y_valid_len)
            train_loss.sum().backward()
            grad_clipping(net, 1)
            optimizer.step()
