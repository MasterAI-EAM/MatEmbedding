import pandas as pd
from scipy.spatial.distance import cosine
from scipy import stats
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

from utils import find_token

'''
there are 5 kinds of bert embedding:
1. context-free-name: only input material name, and directly average token embeddings
2. context-free-CLS: only input material name and special token, use CLS token
3. context-average-name: input list of sentences, average of average(token embeddings of name)
4. context-average-CLS: input list of sentences, use average of CLS token embeddings
5. context-average-sen: input list of sentences, average of average(token embeddings in sentence)
Use single layer extracting, but each layer.
when extracting, 4=2, 5=1, the only difference is input (change from material name to sentence)
so three functions under Extractor: 
1. avg_all (for context-free-name & context-average-sen), 
2. only_CLS (for context-free-CLS & context-average-CLS),
3. avg_need (for context-average-name)
'''


class Extractor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path, do_lower_case=False)
        self.model = BertModel.from_pretrained(self.model_path, output_hidden_states=True).eval()

    def avg_all(self, name, layer_idx):
        tokenized_text = self.tokenizer.tokenize(name)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            # shape: 13, (batch_size, sequence_length, hidden_size)
            hidden_states = outputs.hidden_states
        nth_layer_embedding = hidden_states[layer_idx]
        avg_embedding = torch.mean(nth_layer_embedding, dim=1).squeeze()
        return avg_embedding.numpy()

    def only_CLS(self, name, layer_idx):
        tokens = self.tokenizer.encode(name, add_special_tokens=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(tokens)
            hidden_states = outputs.hidden_states
        cls_embedding = hidden_states[layer_idx][0][0]
        return cls_embedding.numpy()

    def avg_need(self, name, text, layer_idx):
        tokenized_text = self.tokenizer.tokenize(text)
        need_idx = find_token(name, tokenized_text)
        if len(need_idx) == 0:
            # not found name in the text
            return False
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        # remove batch dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # size: x token, 13 layer, 768 hidden unit
        token_embeddings = token_embeddings.permute(1, 0, 2)
        # all token embedding of x layer
        token_vecs_sum = [token[layer_idx] for token in token_embeddings]
        # avg of needed token embeddings
        sentence_embedding = torch.mean(torch.stack([token_vecs_sum[i] for i in need_idx]), 0)
        return sentence_embedding.numpy()

