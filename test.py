import gpt.model as gpt
import gpt.data as data
from vocab import load_vocab
import config as cfg
import argparse
import torch
import sentencepiece as spm
torch.set_printoptions(sci_mode=False)


# 설정, 단어사전 파일 가져오기
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="gpt/config_half.json", type=str, required=False,
                        help="config file")
parser.add_argument("--vocab", default="YT_cmts/Youtube_Comment.model", type=str, required=False,
                        help="vocab file")
parser.add_argument("--pretrain_file", default="YT_cmts/save_pretrain.pth", type=str, required=False,
                        help="save file")
parser.add_argument("--input", default="YT_cmts/YT_cmts_gpt.json", type=str, required=False,
                        help="input pretrain data file")
parser.add_argument("--batch", default=1, type=int, required=False,
                        help="batch") # cpu로 16했더니 팅김
args = parser.parse_args()
args.n_gpu = 0

vocab = spm.SentencePieceProcessor()
vocab = load_vocab(args.vocab)


# config
config = cfg.Config.load(args.config)
config.n_dec_vocab = len(vocab)
config.device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

# 모델, 데이터 로드
model = gpt.GPTPretrain(config)
model.gpt.load(args.pretrain_file, map_location=torch.device('cpu'))
model.eval()

train_loader, train_sampler = data.build_pretrain_loader(vocab, args, shuffle=True)
criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=config.i_pad, reduction='mean')

# 테스트
for i, value in enumerate(train_loader):
    text = input()
    if text!="":
        pieces = vocab.encode_as_pieces(text)
        ids = vocab.encode_as_ids(text)
        print(pieces)
        print(ids)
        dec_inputs = torch.tensor(ids, device=config.device).view(1, len(ids)) 
        print("\n입력: " + vocab.DecodeIdsWithCheck(dec_inputs[0].tolist()))
        print(dec_inputs.shape)
    else: # 입력값, 정답값 가져오기
        dec_inputs, _ = map(lambda v: v.to(config.device), value)
        # labels_lm = dec_inputs[:, 1:]
        labels_lm = dec_inputs[:, 2:]
        dec_inputs = dec_inputs[:, :-1]
        print("\n입력: " + vocab.DecodeIdsWithCheck(dec_inputs[0].tolist()))
        print("\n정답: " + vocab.DecodeIdsWithCheck(labels_lm[0].tolist()))
        print(dec_inputs.shape, labels_lm.shape)
    
    # 신경망 흐르기
    outputs = model(dec_inputs)
    logits_lm = outputs[0]
    logit_lm_0 = logits_lm.view(-1, logits_lm.size(2))
    # print(logits_lm.shape, logit_lm_0.shape)

    # 손실
    if text=="":
        loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
        print("\nLoss: ", round(loss_lm.item(), 4))
    
    # 결과
    chances = torch.softmax(logit_lm_0, 1)
    chances2 = chances.clone().detach()

    for n_rank in range(chances.shape[1]):
        argmax = chances2.argmax(1)
        min = chances2.min(1)[0]

        for i in range(len(chances)):
            chances2[i][argmax[i]] = min[i]

        ids = chances2.argmax(1)
        print(f"\n생성 ({n_rank+1}순위): " + vocab.DecodeIdsWithCheck(ids.tolist()))
        break
        if input() in ['break','exit','/b','/e']: break
        