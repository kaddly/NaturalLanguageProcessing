import torch
from torch import nn
from torch.nn import functional as F
import time
from datetime import timedelta
import math


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


def evaluate_accuracy_gpu(net, data_iter, vocab, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    state_f, state_b = None, None
    loss, fw_l, bw_l = [], [], []
    with torch.no_grad():
        for batch in data_iter:
            seqs, seqs_fw, seqs_bw = [x.to(device) for x in batch]
            if state_f is None:
                # 在第一次使用随机抽样时初始化状态
                state_f = net.module.begin_state(batch_size=seqs.shape[0], device=device)
                state_b = net.module.begin_state(batch_size=seqs.shape[0], device=device)
            else:
                if isinstance(net, nn.Module) and not isinstance(state_f, tuple):
                    # state 对于GRU是个张量
                    state_f.detach_()
                    state_b.detach_()
                else:
                    # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                    for sf, sb in zip(state_f, state_b):
                        sf.detach_()
                        sb.detach_()
            fw_hat, bw_hat, state_f, state_b = net(seqs, state_f, state_b)
            fw_loss = F.cross_entropy(fw_hat.reshape(-1, len(vocab)), seqs_fw.reshape(-1))
            bw_loss = F.cross_entropy(bw_hat.reshape(-1, len(vocab)), seqs_bw.reshape(-1))
            loss.append((fw_loss + bw_loss) / 2)
            fw_l.append(math.exp(fw_loss.mean() * seqs_fw.numel() / seqs_fw.numel()))
            bw_l.append(math.exp(bw_loss.mean() * seqs_bw.numel() / seqs_bw.numel()))
        return sum(loss) / len(loss), sum(fw_l) / len(fw_l), sum(bw_l) / len(bw_l)


def train(net, train_iter, test_iter, num_epochs, lr, devices, vocab, use_random_iter):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(init_weights)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    loss = nn.CrossEntropyLoss()
    start_time = time.time()
    net.train()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    for epoch in range(num_epochs):
        state_f, state_b = None, None
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            seqs, seqs_fw, seqs_bw = [x.to(devices[0]) for x in batch]
            if state_f is None or use_random_iter:
                # 在第一次使用随机抽样时初始化状态
                state_f = net.module.begin_state(batch_size=seqs.shape[0], device=devices[0])
                state_b = net.module.begin_state(batch_size=seqs.shape[0], device=devices[0])
            else:
                if isinstance(net, nn.Module) and not isinstance(state_f, tuple):
                    # state 对于GRU是个张量
                    state_f.detach_()
                    state_b.detach_()
                else:
                    # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                    for sf, sb in zip(state_f, state_b):
                        sf.detach_()
                        sb.detach_()
            fw_hat, bw_hat, state_f, state_b = net(seqs, state_f, state_b)
            fw_loss = loss(fw_hat.reshape(-1, len(vocab)), seqs_fw.reshape(-1))
            bw_loss = loss(bw_hat.reshape(-1, len(vocab)), seqs_bw.reshape(-1))
            l = (fw_loss + bw_loss) / 2
            l.backward()
            grad_clipping(net, 1)
            optimizer.step()
            if total_batch % 50 == 0:
                dev_loss, dev_f_loss, dev_b_loss = evaluate_accuracy_gpu(net, test_iter, vocab)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    improve = '*'
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:>6},  T_loss: {1:>5.3f},  TF_Perplexity: {2:>5.3f},  TB_Perplexity: {3:>5.3f},  TF_Perplexity: {4:>5.3f},  TB_Perplexity: {5:>5.3f},  Time: {6} {7}'
                with torch.no_grad():
                    print(msg.format(total_batch, l.mean(),
                                     math.exp(fw_loss.mean() * seqs_fw.numel() / seqs_fw.numel()),
                                     math.exp(bw_loss.mean() * seqs_bw.numel() / seqs_bw.numel()), dev_f_loss,
                                     dev_b_loss,
                                     time_dif, improve))
                    print("| tgt:", " ".join(vocab.to_tokens(seqs[0].cpu().tolist())),
                          "\n| f_prd:", " ".join(vocab.to_tokens(torch.argmax(fw_hat[0], axis=1).cpu().tolist())),
                          "\n| b_prd:", " ".join(vocab.to_tokens(torch.argmax(bw_hat[0], axis=1).cpu().tolist())),
                          "\n")
                torch.save(net.state_dict(), './saved_dict/BiRNN.ckpt')
                net.train()
            total_batch += 1
