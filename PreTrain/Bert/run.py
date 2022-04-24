import torch
from torch import nn
from utils import load_data_wiki
from Bert import BERTModel
from train_eval import train_bert, get_bert_encoding


if __name__ == '__main__':
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    net = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128], ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128, value_size=128, hid_in_features=128,
                    mlm_in_features=128, nsp_in_features=128)
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
    loss = nn.CrossEntropyLoss()
    train_bert(train_iter, net, loss, len(vocab), devices, 50)
    tokens_a = ['a', 'crane', 'is', 'flying']
    encoded_text = get_bert_encoding(net, vocab, devices, tokens_a)
    # 词元：'<cls>','a','crane','is','flying','<sep>'
    encoded_text_cls = encoded_text[:, 0, :]
    encoded_text_crane = encoded_text[:, 2, :]
    print(encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3])
    # BERT表⽰是上下⽂敏感的
    tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
    encoded_pair = get_bert_encoding(net, vocab, devices, tokens_a, tokens_b)
    # 词元：'<cls>','a','crane','driver','came','<sep>','he','just', 'left','<sep>'
    encoded_pair_cls = encoded_pair[:, 0, :]
    encoded_pair_crane = encoded_pair[:, 2, :]
    print(encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3])

