import torch
from torch import nn
import time
from datetime import timedelta


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


def evaluate_accuracy_gpu(net, data_iter, token_loss, fineTurn, theta, device=None):
    pass


def train(net, train_iter, test_iter, num_epochs, lr, devices, vocab, use_random_iter):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    loss = nn.CrossEntropyLoss()
    start_time = time.time()
    net.train()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    for epoch in range(num_epochs):
        state = None
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            seqs, seqs_fw, seqs_bw = [x.to(devices[0]) for x in batch]
            if state is None or use_random_iter:
                # 在第一次使用随机抽样时初始化状态
                state = net.begin_state(batch_size=seqs.shape[0], device=devices[0])
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # state 对于GRU是个张量
                    state.detach_()
                else:
                    # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                    for s in state:
                        s.detach_()
            fw_hat, bw_hat, state = net(seqs, state)
            l = (
                        loss(fw_hat.reshape(-1, len(vocab)), seqs_fw.reshape(-1)) +
                        loss(bw_hat.reshape(-1, len(vocab)), seqs_bw.reshape(-1))
                ) / 2
            l.backward()
            grad_clipping(net, 1)
            optimizer.step()
            if total_batch % 50 == 0:
                dev_loss = evaluate_accuracy_gpu(net, test_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(net.state_dict(), './saved_dict/BiRNN.ckpt')
                    improve = '*'
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, l, dev_loss, time_dif, improve))
                net.train()
            total_batch += 1
