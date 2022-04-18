import torch
from torch import nn
import time


class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super.__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)


def train(net, data_iter, lr, num_epochs, device):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(params=net.net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        tik = time.time()
        num_batches = len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = net(center, context_negative)
            l = (SigmoidBCELoss(pred.reshape(label.shape).float(), label.float(), mask)/mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            with torch.no_grad():
                l_sum = l.sum()
                l_nums = l.numel()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f"step:{epoch + (i+1)/num_batches}>>loss:{l_sum/ l_nums:.3f}")
    print(f'loss {l_sum / l_nums:.3f}, {l_nums / (time.time() - tik):.1f} 'f'tokens/sec on {str(device)}')

