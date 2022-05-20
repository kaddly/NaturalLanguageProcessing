import torch
from .ELMO import ELMO
from .data_utils import load_WikiTextDataset
from .train_eval import train

if __name__ == '__main__':
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
    batch_size, num_epochs = 256, 100
    num_steps, lr = 32, 2e-3
    train_iter, valid_iter, test_iter, vocab = load_WikiTextDataset(batch_size, num_steps)
    net = ELMO(len(vocab), embedding_size=256, hidden_size=256, num_layers=2)
    train(net, train_iter, valid_iter, num_epochs, lr, devices, vocab, False)
