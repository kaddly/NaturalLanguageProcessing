import torch
from torch.utils.data import DataLoader
from Transformer import TransformerEncoder, TransformerDecoder, EncoderDecoder
from utils import My_Dataset, show_heatmaps
from train_eval import train_seq2seq, predict_seq2seq, bleu
import pandas as pd

if __name__ == '__main__':
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 320, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    my_dataset = My_Dataset(num_steps=num_steps)
    src_vocab, tgt_vocab = my_dataset.get_vocab()
    train_iter = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
                                 ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
                                 ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    # 预测
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ', f'bleu {bleu(translation, fra, k=2):.3f}')
    enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads, -1, num_steps))
    show_heatmaps(enc_attention_weights.cpu(), xlabel='Key positions', ylabel='Query positions',
                  titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
    dec_attention_weights_2d = [head[0].tolist()
                                for step in dec_attention_weight_seq
                                for attn in step for blk in attn for head in blk]
    dec_attention_weights_filled = torch.tensor(pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
    dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
    dec_self_attention_weights, dec_inter_attention_weights = dec_attention_weights.permute(1, 2, 3, 0, 4)
    show_heatmaps(dec_self_attention_weights[:, :, :, :len(translation.split()) + 1], xlabel='Key positions',
                  ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
