from LSTM import RNNModel
from utils import load_data_time_machine
from train_eval import train_seq
import torch
import torch.nn as nn

if __name__ == '__main__':
    # batch_size, num_steps = 32, 35
    # num_epochs, lr = 500, 1
    # train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    # num_hiddens = 256
    # lstm_layer = nn.LSTM(len(vocab), num_hiddens)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = RNNModel(lstm_layer, vocab_size=len(vocab))
    # net = net.to(device)
    # train_seq(net, train_iter, vocab, lr, num_epochs, device)
    # 深度循环神经网络
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    model = RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    num_epochs, lr = 500, 2
    train_seq(model, train_iter, vocab, lr, num_epochs, device)
