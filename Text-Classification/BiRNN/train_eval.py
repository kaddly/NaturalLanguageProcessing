import torch
from torch import nn
import time


def train(model, train_iter, dev_iter, test_iter, loss, devices, lr, num_epochs):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            trains = trains.to(devices[0])
            labels = labels.to(devices[0])
            outputs = model(trains)
            model.zero_grad()
            l = loss(outputs, labels)
            l.backword()
            optimizer.step()
            if total_batch % 50 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
