import torch
from torch import nn
from GPT import sequence_mask
import time


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


def train_GPT(net, train_iter, test_iter, num_epochs, fineTurn, lr, devices, theta=0.2):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(init_weights)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    token_loss = MaskedSoftmaxCELoss()
    nsp_loss = nn.CrossEntropyLoss()
    start_time = time.time()
    net.train()
    net.fineTurn = fineTurn
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            tokens, segments, valid_lens, labels = [x.to(devices[0]) for x in batch]
            y_hat, y_labels = net(tokens[:, :-1], segments[:, :-1])
            if fineTurn:
                token_l = token_loss(y_hat, tokens[:, 1:], valid_lens)
                nsp_l = nsp_loss(y_labels, labels)
                l = token_l + theta * nsp_l
            else:
                l = token_loss(y_hat, tokens[:, 1:], valid_lens)
            l.sum().brackward()
            grad_clipping(net, 1)
            optimizer.step()
