import torch
from word2vec import Word2vec
from utils import load_data_ptb
from train_eval import train, get_similar_tokens

if __name__ == '__main__':
    batch_size, max_window_size, num_noise_words = 512, 5, 5
    lr, num_epochs, device = 0.002, 5, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_iter, vocab = load_data_ptb(batch_size, max_window_size, num_noise_words)
    vocab_size, embed_size = len(vocab), 100
    model = Word2vec(vocab_size, embed_size)
    train(model, data_iter, lr, num_epochs, device)
    get_similar_tokens('chip', 3, model.net[0], vocab)
