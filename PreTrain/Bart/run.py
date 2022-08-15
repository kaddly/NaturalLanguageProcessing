import torch
from models import BartEncoder, BartDecoder, EncoderDecoder
from utils import load_data_wiki, show_heatmaps
from train_utils import train, test, bleu

if __name__ == '__main__':
    num_hiddens, num_layers, dropout = 64, 4, 0.1
    ffn_num_input, ffn_num_hiddens, num_heads = 64, 128, 8
    batch_size, max_len, num_merge = 512, 64, 10000
    lr, num_epoch = 0.05, 100
    key_size, query_size, value_size = 64, 64, 64
    reconstruct_ways = ['Sentence_Permutation', 'Text_Infilling']
    norm_shape = [64]
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
    train_iter, val_iter, test_iter, vocab = load_data_wiki(batch_size, max_len, reconstruct_ways, num_merge)
    encoder = BartEncoder(len(vocab), key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = BartDecoder(len(vocab), key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train(net, train_iter, val_iter, lr, num_epoch, vocab, devices)
