# -*- coding: utf-8 -*-
# @Time    : 2022/9/9 10:54
# @Author  : WAN Yuwei
# @FileName: calculate_avg_score.py
# @Email: yuweiwan2-c@my.cityu.edu.hk
# @Github: https://github.com/yuweiwan
# @Personal Website: https://yuweiwan.github.io/
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import json
from tqdm import tqdm
import nltk
from random import sample


def find_token(mat, tok_sen):
    candidate = []
    for i, t in enumerate(tok_sen):
        if mat.startswith(t):
            idx = [i]
            can = t
            tmp = i
            if can == mat:
                candidate.extend(idx)
                break
            if i == len(tok_sen) - 1:
                break
            while True:
                can += tok_sen[tmp + 1].replace('##', '')
                if not mat.startswith(can):
                    break
                idx.append(tmp + 1)
                if can == mat:
                    candidate.extend(idx)
                    break
                if tmp == len(tok_sen) - 2:
                    break
                else:
                    tmp += 1
    return candidate


def get_embedding1(text0, tokenizer, model):
    tokenized_text0 = tokenizer(text0)  # 将输入按照BERT的处理方式进行分割
    segments_ids = [1] * len(tokenized_text0)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text0)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():  # 将输入传入模型，得到每一层的输出信息，这里的encoded_layers为12层，可以打印验证
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding


def get_embedding2(mat, text0, tokenizer, model):
    tokenized_text = tokenizer.tokenize(text0)
    # print([mat], tokenized_text)
    need_idx = find_token(mat, tokenized_text)
    # print(need_idx)
    if not need_idx:
        return None
    # reduce length
    if len(tokenized_text) > 512:
        if need_idx[-1] < 512:
            tokenized_text = tokenized_text[:512]
        else:
            return None
        '''
        elif len(tokenized_text) - need_idx[0] < 512:
            tokenized_text = tokenized_text[len(tokenized_text) - 512:]
            tmp = need_idx
            need_idx = []
            for i in tmp:
                need_idx.append(i - len(tokenized_text) + 512)
        else:
            return None
        '''
    segments_ids = [1] * len(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():  # 将输入传入模型，得到每一层的输出信息，这里的encoded_layers为12层，可以打印验证
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)
        embeddings = []
        for i in need_idx:
            embeddings.append(token_vecs_sum[i])
        sentence_embedding = torch.mean(torch.stack(embeddings), 0)
        return sentence_embedding


tokenizer = BertTokenizer.from_pretrained('mat_bert_cased', do_lower_case=False)
model = BertModel.from_pretrained('mat_bert_cased', output_hidden_states=True).eval()
sen = 'passivation, conductivity, and selectivity are often acknowledged as the three requirements for optimal ' \
      'contacts to photovoltaic solar cells '
v1 = get_embedding2('conductivity', sen, tokenizer, model)
v2 = get_embedding2('passivation', sen, tokenizer, model)
v3 = get_embedding2('selectivity', sen, tokenizer, model)
# v4 = get_embedding2('contacts', sen, tokenizer, model)
# v_test = get_embedding2('contactpassivation', 'contact passivation conductivity', tokenizer, model)
v = torch.mean(torch.stack([v1, v2, v3]), 0)
# v_sen = get_embedding1(sen, tokenizer, model)
score_dict = {}
tensors = torch.load('tensors.pt')
for i in tensors.keys():
    score_dict[i] = 1 - cosine(v, tensors[i])

sorted_score = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
json_str = json.dumps(sorted_score, indent=4)
with open('sorted_score_sen.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_str)
