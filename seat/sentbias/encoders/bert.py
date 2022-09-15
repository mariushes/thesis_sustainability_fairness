''' Convenience functions for handling BERT '''
import torch
import pytorch_pretrained_bert as bert


import transformers

def load_model_transformers(version):
    tokenizer = transformers.BertTokenizer.from_pretrained(version)
    model = transformers.BertModel.from_pretrained(version)
    model.eval()
    
    return model, tokenizer

def load_model(version='bert-large-uncased'):
    ''' Load BERT model and corresponding tokenizer '''
    tokenizer = bert.BertTokenizer.from_pretrained(version)
    model = bert.BertModel.from_pretrained(version)
    model.eval()

    return model, tokenizer

def encode_transformers(model, tokenizer, device, texts):
    encs = {}
    for text in texts:
        tokens_tensor =tokenizer.encode(text, return_tensors="pt")
        #print(tokens_tensor)
        #print(type(tokens_tensor))
        #print(len(tokens_tensor))
        #print(tokens_tensor.size())
        segment_idxs = [0] * len(tokens_tensor)
        segments_tensor = torch.tensor([segment_idxs])
        output = model(input_ids=tokens_tensor.to(device), token_type_ids=segments_tensor.to(device), output_hidden_states=True)
        enc = output.hidden_states[-1]
        enc = enc[:, 0, :]
        #print(enc.size())
        encs[text] = enc.detach().cpu().view(-1).numpy()
    return encs
    

def encode(model, tokenizer, texts):
    ''' Use tokenizer and model to encode texts '''
    encs = {}
    for text in texts:
        print("Text: ",text )
        tokenized = tokenizer.tokenize(text)
        indexed = tokenizer.convert_tokens_to_ids(tokenized)
        segment_idxs = [0] * len(tokenized)
        tokens_tensor = torch.tensor([indexed])
        segments_tensor = torch.tensor([segment_idxs])
        enc, _ = model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)
        print("Enc Pre slice")
        print(enc.size())
        print(enc)
        enc = enc[:, 0, :]  # extract the last rep of the first input
        print("Enc Post slice")
        print(enc.size())
        print(enc)
        encs[text] = enc.detach().view(-1).numpy()
    return encs

