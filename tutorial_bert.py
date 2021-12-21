import torch
import sentencepiece as spm
from bert.model import MovieClassification, BERTPretrain
from bert.data import MovieDataSet, movie_collate_fn, PretrainDataSet, make_pretrain_data, pretrin_collate_fn
from config import Config

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd

from modules import enc_input_to_sentence, test_one
import os

################################################################################################################################

# vocab loading
data_dir = "data"
vocab_file = f"{data_dir}/kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

################################################################################################################################

""" 데이터 로더 """
batch_size = 128
train_dataset = MovieDataSet(vocab, f"{data_dir}/ratings_train.json")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=movie_collate_fn)
test_dataset = MovieDataSet(vocab, f"{data_dir}/ratings_test.json")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=movie_collate_fn)


# for labels, enc_inputs, dec_inputs in train_loader:
    # print(labels)
    # print(enc_inputs)
    # print(dec_inputs)
    # print(labels.shape, enc_inputs.shape, dec_inputs.shape)
    # for i in range(10):
    #     print(vocab.IdToPiece(i), end=" ")
    # print()
        
    # for idx in range(len(enc_inputs)):
    #     sentence = enc_input_to_sentence(enc_inputs, idx)
    #     input(sentence)
    # break

################################################################################################################################

""" 모델 epoch 학습 """
def train_epoch(config, epoch, model, criterion_cls, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
        for i, value in enumerate(train_loader):
            labels, inputs, segments = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            outputs = model(inputs, segments)
            logits_cls = outputs[0]

            loss_cls = criterion_cls(logits_cls, labels)
            loss = loss_cls

            loss_val = loss_cls.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

""" 모델 epoch 평가 """
def eval_epoch(config, model, data_loader):
    matchs = []
    model.eval()

    n_word_total = 0
    n_correct_total = 0
    with tqdm(total=len(data_loader), desc=f"Valid") as pbar:
        for i, value in enumerate(data_loader):
            labels, inputs, segments = map(lambda v: v.to(config.device), value)

            outputs = model(inputs, segments)
            logits_cls = outputs[0]
            _, indices = logits_cls.max(1)

            match = torch.eq(indices, labels).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
    return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

################################################################################################################################

config = Config({
    "n_enc_vocab": len(vocab),
    "n_enc_seq": 256,
    "n_seg_type": 2,
    "n_layer": 2, # 6 (default value)
    "d_hidn": 256,
    "i_pad": 0,
    "d_ff": 512, # 1024 (default value)
    "n_head": 4,
    "d_head": 64,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-12
})
print(config)

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.n_output = 2

learning_rate = 5e-5
n_epoch = 1 # 10 (default value)

################################################################################################################################

model = BERTPretrain(config)

save_pretrain = f"{data_dir}/save_bert_pretrain.pth"
best_epoch, best_loss = 0, 0
if os.path.isfile(save_pretrain):
    best_epoch, best_loss = model.bert.load(save_pretrain, map_location=config.device)
    print(f"load pretrain from: {save_pretrain}, epoch={best_epoch}, loss={best_loss}")
    best_epoch += 1

model.to(config.device)

criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
criterion_cls = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
offset = best_epoch
for step in range(n_epoch):
    epoch = step + offset
    if 0 < step:
        del train_loader
        dataset = PretrainDataSet(vocab, f"{data_dir}/kowiki_bert_{epoch % count}.json")
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pretrin_collate_fn)

    loss = train_epoch(config, epoch, model, criterion_lm, criterion_cls, optimizer, train_loader)
    losses.append(loss)
    model.bert.save(epoch, loss, save_pretrain)

################################################################################################################################

def train(model):
    model.to(config.device)

    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_epoch, best_loss, best_score = 0, 0, 0
    losses, scores = [], []
    for epoch in range(n_epoch):
        loss = train_epoch(config, epoch, model, criterion_cls, optimizer, train_loader)
        score = eval_epoch(config, model, test_loader)

        losses.append(loss)
        scores.append(score)

        if best_score < score:
            best_epoch, best_loss, best_score = epoch, loss, score
    print(f">>>> epoch={best_epoch}, loss={best_loss:.5f}, socre={best_score:.5f}")
    return losses, scores


# model = MovieClassification(config)
# losses_00, scores_00 = train(model)

model = MovieClassification(config)
model.bert.load(save_pretrain, map_location=config.device)
losses_20, scores_20 = train(model)

################################################################################################################################

try:
    model.save("bert_tutorial.pth")
except:
    torch.save(model.state_dict(), "bert_tutorial.pth")

################################################################################################################################

test_one(MovieClassification, config, test_loader, "bert_tutorial.pth")

# # table
# data = {
#     "loss_00": losses_00,
#     "socre_00": scores_00,
#     "loss_20": losses_20,
#     "socre_20": scores_20,
# }
# df = pd.DataFrame(data)
# display(df)

# # graph
# plt.figure(figsize=[12, 4])
# plt.plot(scores_00, label="score_00")
# plt.plot(scores_20, label="score_20")
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.show()
