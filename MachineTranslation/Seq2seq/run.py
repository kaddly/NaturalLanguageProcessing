import torch
from torch.utils.data import DataLoader
from Seq2seq import Seq2SeqEncoder, Seq2SeqDecoder, EncoderDecoder
from utils import My_Dataset
from train_eval import train_seq2seq, predict_seq2seq, bleu

if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_dataset = My_Dataset(num_steps=num_steps)
    src_vocab, tgt_vocab = my_dataset.get_vocab()
    train_iter = DataLoader(my_dataset, batch_size=batch_size)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 测试模型
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
