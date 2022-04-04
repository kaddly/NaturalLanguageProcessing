from RNN import *
from utils import *
from train_eval import *
import torch
import torch.nn as nn

if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    num_epochs, lr = 500, 1
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)
    train_seq(net,train_iter,vocab,lr,num_epochs,device)