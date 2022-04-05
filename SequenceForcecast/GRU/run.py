from GRU import RNNModel
from utils import load_data_time_machine
from train_eval import train_seq
import torch
import torch.nn as nn

if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    num_epochs, lr = 500, 1
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    gru_layer = nn.GRU(len(vocab), num_hiddens)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RNNModel(gru_layer, vocab_size=len(vocab))
    net = net.to(device)
    train_seq(net, train_iter, vocab, lr, num_epochs, device)
