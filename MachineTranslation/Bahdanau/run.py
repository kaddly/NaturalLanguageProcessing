import torch
from torch.utils.data import DataLoader
from Bahdanau import Seq2SeqEncoder, Seq2SeqAttentionDecoder, EncoderDecoder
from utils import My_Dataset, show_heatmaps
from train_eval import train_seq2seq, predict_seq2seq, bleu

if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_dataset = My_Dataset(num_steps=num_steps)
    src_vocab, tgt_vocab = my_dataset.get_vocab()
    train_iter = DataLoader(my_dataset, batch_size=batch_size,shuffle=False)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 测试模型
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
              f'bleu {bleu(translation, fra, k=2):.3f}')

    # 可视化注意力
    attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape(
        (1, 1, -1, num_steps))
    show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
                  xlabel='Key positions', ylabel='Query positions')
