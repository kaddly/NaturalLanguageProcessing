import torch
import pandas as pd
from models import BartEncoder, BartDecoder, EncoderDecoder
from utils import load_data_wiki, show_heatmaps
from train_utils import train, test, bleu, predict

if __name__ == '__main__':
    num_hiddens, num_layers, dropout = 64, 2, 0.1
    ffn_num_input, ffn_num_hiddens, num_heads = 64, 128, 4
    batch_size, max_len, num_merge = 128, 64, 5000
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
    test(net, test_iter, devices, vocab)
    # 预测
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    tages = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    for eng, tag in zip(engs, tages):
        translation, dec_attention_weight_seq = predict(net, eng, vocab, max_len, devices, True)
        print(f'{eng} => {translation}, ', f'bleu {bleu(translation, tag, k=2):.3f}')
    enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads, -1, max_len))
    show_heatmaps(enc_attention_weights.cpu(), xlabel='Key positions', ylabel='Query positions',
                  titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
    dec_attention_weights_2d = [head[0].tolist()
                                for step in dec_attention_weight_seq
                                for attn in step for blk in attn for head in blk]
    dec_attention_weights_filled = torch.tensor(pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
    dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, max_len))
    dec_self_attention_weights, dec_inter_attention_weights = dec_attention_weights.permute(1, 2, 3, 0, 4)
    show_heatmaps(dec_self_attention_weights[:, :, :, :len(translation.split()) + 1], xlabel='Key positions',
                  ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
