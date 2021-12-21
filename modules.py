def enc_input_to_sentence(enc_inputs, idx, vocab):
    sentence = []
    for idx2 in range(len(enc_inputs[idx])):
        id = int(enc_inputs[idx][idx2])
        if vocab.IdToPiece(id) == "[PAD]": break
        sentence.append(vocab.IdToPiece(id))
        
    return " ".join(sentence)+"\n"



def test_one_by_one(model_class, config, vocab, test_loader, saved_model="model_tutorial.pth"):
    model = model_class(config)
    model.load(saved_model)
    model.to(config.device)
    
    for value in test_loader:
        labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)
        outputs = model(enc_inputs, dec_inputs)
        logits = outputs[0]
        _, indices = logits.max(1)

        # print(outputs[1][0].shape, outputs[2][0].shape, outputs[3][0].shape)

        for idx in range(len(enc_inputs)):
            predict = "긍정" if indices[idx].item()==1 else "부정"
            answer = "긍정" if labels[idx].item()==1 else "부정"
            print("예측: " + predict + "\n정답: " + answer)
            input("문장: " + vocab.DecodeIdsWithCheck(enc_inputs[idx].tolist()))