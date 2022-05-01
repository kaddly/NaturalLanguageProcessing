import torch
from torch import nn
from torch.utils.data import DataLoader
from BERTLoadPreTrain import load_pretrained_model, BERTClassifier
from data_utils import read_snli, SNLIBERTDataset
from train_eval import train_snli, predict_snli

if __name__ == '__main__':
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
    bert, vocab = load_pretrained_model(
        'bert.small.torch', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
        num_layers=2, dropout=0.1, max_len=512, devices=devices)
    # 如果出现显存不⾜错误，请减少“batch_size”。在原始的BERT模型中，max_len=512
    batch_size, max_len, num_workers = 512, 128, 2
    data_dir = '../data/snli_1.0/'
    train_set = SNLIBERTDataset(read_snli(data_dir, True), max_len, vocab)
    test_set = SNLIBERTDataset(read_snli(data_dir, False), max_len, vocab)
    train_iter = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(test_set, batch_size, num_workers=num_workers)
    net = BERTClassifier(bert)
    lr, num_epochs = 1e-4, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    train_snli(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
