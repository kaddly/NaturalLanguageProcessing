import torch
from data_utils import load_data_MSRPC
from GPT import GPTModel
from train_eval import train_GPT

if __name__ == '__main__':
    num_epochs, batch_size = 20, 32
    lr, dropout = 1e-4, 0.2
    model_dim, num_layers, num_heads = 256, 4, 4
    train_iter, test_iter, vocab = load_data_MSRPC(batch_size, max_len=64)
    net = GPTModel(len(vocab), num_hiddens=model_dim, norm_shape=[model_dim], ffn_num_input=model_dim,
                   ffn_num_hiddens=2 * model_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout,
                   max_len=1000, key_size=model_dim, query_size=model_dim, value_size=model_dim,
                   hid_in_features=model_dim, nsp_in_features=model_dim, fineTurn=True)
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
    train_GPT(net, train_iter, test_iter, num_epochs, True, lr, devices, vocab, theta=0.2)
