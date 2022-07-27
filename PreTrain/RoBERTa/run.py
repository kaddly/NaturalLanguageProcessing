import torch
from models import RoBERTaModel
from .data_utils import load_wiki
from train_utils import train, test

if __name__ == '__main__':
    batch_size, max_len, num_merge = 512, 64, 10000
    train_iter, val_iter, test_iter, BPE = load_wiki(batch_size, max_len, num_merge)
    net = RoBERTaModel(len(BPE), num_hiddens=128, norm_shape=[128], ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                       num_layers=2, dropout=0.2, key_size=128, query_size=128, value_size=128, mlm_in_features=128)
    print(net)
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
