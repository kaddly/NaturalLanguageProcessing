import time
import os
import torch
from torch import nn
from datetime import timedelta
from .scale_utils import accuracy, MaskedSoftmaxCELoss
from .optimizer_utils import create_lr_scheduler, grad_clipping
from .distributed_utils import Accumulator


def evaluate_accuracy_gpu(net, data_iter, loss, vocab, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    with torch.no_grad():
        val_acc, val_loss = [], []
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([vocab['<s>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            val_acc.append(accuracy(Y_hat.reshape(-1, len(vocab)), Y.reshape(-1)))
            val_loss.append(loss(Y_hat, Y, Y_valid_len).mean())
    return sum(val_acc) / len(val_acc), sum(val_loss) / len(val_loss)


def train(net, train_iter, val_iter, lr, num_epochs, vocab, devices, is_current_train=True):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    # 模型参数保存路径
    saved_dir = './saved_dict'
    model_file = 'Bart'
    parameter_path = os.path.join(saved_dir, model_file)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    if is_current_train and os.path.exists('./saved_dict/Bart/Bart.ckpt'):
        net.load_state_dict(torch.load('./saved_dict/Bart/Bart.ckpt'), False)
    else:
        net.apply(init_weights)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_iter), num_epochs)
    loss = MaskedSoftmaxCELoss()
    start_time = time.time()

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    metric = Accumulator(3)

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(devices[0]) for x in batch]
            bos = torch.tensor([vocab['<s>']] * Y.shape[0], device=devices[0]).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            train_loss = loss(Y_hat, Y, Y_valid_len)
            train_loss.sum().backward()
            grad_clipping(net, 1)
            optimizer.step()
            lr_scheduler.step()
            with torch.no_grad():
                metric.add(train_loss.sum(), accuracy(Y_hat.reshape(-1, len(vocab)), Y.reshape(-1)), Y.shape[0])
            if total_batch % 20 == 0:
                lr_current = optimizer.param_groups[0]["lr"]
                dev_acc, dev_loss = evaluate_accuracy_gpu(net, val_iter, loss, vocab)
                if dev_loss < dev_best_loss:
                    torch.save(net.state_dict(), os.path.join(parameter_path, model_file + '.ckpt'))
                    dev_best_loss = dev_loss
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.4},  Train Acc: {2:>6.2%},  Train lr: {3:>5.4},  Val Loss: {4:>5.4},  Val Acc: {5:>6.2%},  Time: {6} {7}'
                print(
                    msg.format(total_batch, metric[0] / metric[2], metric[1] / (total_batch + 1), lr_current, dev_loss,
                               dev_acc, time_dif, improve))
                net.train()
            total_batch += 1
            if total_batch - last_improve > 5000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test(model, data_iter, devices, vocab):
    if not os.path.exists('./saved_dict/Bart/Bart.ckpt'):
        print('please train before!')
        return
    model.load_state_dict(torch.load('./saved_dict/Bart/Bart.ckpt'), False)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    model.eval()
    loss = MaskedSoftmaxCELoss()
    with torch.no_grad():
        test_acc, test_loss = [], []
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(devices[0]) for x in batch]
            bos = torch.tensor([vocab['<s>']] * Y.shape[0], device=devices[0]).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = model(X, dec_input, X_valid_len)
            test_acc.append(accuracy(Y_hat.reshape(-1, len(vocab)), Y.reshape(-1)))
            test_loss.append(loss(Y_hat, Y, Y_valid_len).mean())
    print("Test set results:", "loss= {:.4f}".format(sum(test_loss) / len(test_loss)),
          "accuracy= {:.4f}".format(sum(test_acc) / len(test_acc)))


def predict(net, src_sentence, vocab, num_steps, devices, save_attention_weights=False):
    net.eval()
    src_tokens = [vocab['<s>']] + vocab[src_sentence.lower().split(' ')] + [vocab['</s>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=devices[0])
    if num_steps > enc_valid_len:
        src_tokens += [vocab['<pad>']] * (num_steps - enc_valid_len)
    else:
        src_tokens = src_tokens[:num_steps]
        enc_valid_len = num_steps
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=devices[0]), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([vocab['<s>']], dtype=torch.long, device=devices[0]), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == vocab['</s>']:
            break
        output_seq.append(pred)
    return ' '.join(vocab.to_tokens(output_seq)), attention_weight_seq
