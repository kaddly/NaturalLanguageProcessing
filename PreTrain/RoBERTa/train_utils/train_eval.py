import torch
from torch import nn
import torch.nn.functional as f
import time
import os
from datetime import timedelta
from .scale_utils import Accumulator, accuracy


def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def evaluate_accuracy_gpu(net, data_iter, vocab_size, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(3)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, tuple):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            _, y_hat = net(X[0], None, X[1])
            metric.add(f.cross_entropy(y_hat.reshape(-1, vocab_size), y.reshape(-1)),
                       accuracy(y_hat.shape(-1, vocab_size), y.reshape(-1)), 1)
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, val_iter, lr, num_epochs, vocab_size, devices):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, betas=(0.9, 0.98))
    lr_scheduler = create_lr_scheduler(optimizer, len(train_iter), num_epochs)
    loss = nn.CrossEntropyLoss()
    start_time = time.time()

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    metric = Accumulator(4)

    # 模型参数保存路径
    saved_dir = './saved_dict'
    model_file = 'RoBERTa'
    parameter_path = os.path.join(saved_dir, model_file)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (X, labels) in enumerate(train_iter):
            optimizer.zero_grad()
            if isinstance(X, tuple):
                # Required for BERT fine-tuning (to be covered later)
                X = [x.to(devices[0]) for x in X]
            else:
                X = X.to(devices[0])
            labels = labels.to(devices[0])
            encoded_X, mlm_y_hat = net(X[0], None, X[1])
            train_loss = loss(mlm_y_hat.reshape(-1, vocab_size), labels.reshape(-1))
            train_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            with torch.no_grad():
                metric.add(train_loss * labels.shape[0],
                           accuracy(mlm_y_hat.reshape(-1, vocab_size), labels.reshape(-1)), labels.shape[0])
            if total_batch % 20 == 0:
                lr_current = optimizer.param_groups[0]["lr"]
                dev_acc, dev_loss = evaluate_accuracy_gpu(net, val_iter, vocab_size)
                if dev_loss < dev_best_loss:
                    torch.save(net.state_dict(), os.path.join(parameter_path, model_file + '.ckpt'))
                    dev_best_loss = dev_loss
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.4},  Train Acc: {2:>6.2%},  Train lr: {3:>5.4},  Val Loss: {4:>5.4},  Val Acc: {5:>6.2%},  Time: {6} {7}'
                print(msg.format(total_batch, metric[0] / metric[2], metric[1] / total_batch, lr_current, dev_loss,
                                 dev_acc, time_dif, improve))
                net.train()
            total_batch += 1
            if total_batch - last_improve > 1000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test():
    pass
