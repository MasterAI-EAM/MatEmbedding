# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 17:24
# @Author  : WAN Yuwei
# @FileName: try_avg.py
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


def sen_seg(data):
    to_replace = ['et al. ', 'Fig. ', 'e.g. ', 'i.e. ', 'Ref. ', 'Figs. ', ' ca. ', 'approx. ', '(ca. ', 'etc.) ']
    for tr in to_replace:
        data = data.replace(tr, tr[:-2] + '####@')
    tmp = nltk.sent_tokenize(data)
    for i, t in enumerate(tmp):
        for tr in to_replace:
            t = t.replace(tr[:-2] + '####@', tr)
        tmp[i] = t
    return tmp


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


def get_embedding(mat, text0, tokenizer, model):
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
        try:
            sentence_embedding = torch.mean(torch.stack(embeddings), 0)
            return sentence_embedding
        except:
            print('problem mat_name: ', mat)
            return None


# with open('../passivating_abstract.json', 'r', encoding='utf-8') as f:
# abstracts = json.load(f)
# with open('../xml_dict.json', 'r', encoding='utf-8') as f:
# xml_dict = json.load(f)
with open('../pair_list.json', 'r', encoding='utf-8') as f:
    pair_list = json.load(f)
with open('../ori_dict.json', 'r', encoding='utf-8') as f:
    ori_dict = json.load(f)

detection = []
norm_dict = {}
max_dict = {}
for p in pair_list:
    for f in p['full']:
        norm_dict[f] = p['max_abb']
    for f in p['abb']:
        norm_dict[f] = p['max_abb']
    max_dict[p['max_abb']] = p['full']
    max_dict[p['max_abb']].extend(p['abb'])
    detection.extend(p['full'])
    detection.extend(p['abb'])

for o in ori_dict.keys():
    detection.extend(ori_dict[o])
    for ori in ori_dict[o]:
        norm_dict[ori] = o

bran_dict = {}
for n in norm_dict.keys():
    if norm_dict[n] not in bran_dict.keys():
        bran_dict[norm_dict[n]] = [n]
    else:
        bran_dict[norm_dict[n]].append(n)

with open('mat_sentences.json', 'r', encoding='utf-8') as f:
    mat_sentences = json.load(f)

'''
detection = list(set(detection))
mat_sentences = {}
for a in tqdm(abstracts.values()):
    sens = sen_seg(a)
    for s in sens:
        for d in detection:
            if ' '+d+' ' in s or '('+d+')' in s:
                if d not in mat_sentences:
                    mat_sentences[d] = [s]
                else:
                    mat_sentences[d].append(s)
for x in tqdm(xml_dict.keys()):
    if 'paras' in xml_dict[x].keys():
        for p in xml_dict[x]['paras']:
            sens = sen_seg(p)
            for s in sens:
                for d in detection:
                    if ' ' + d + ' ' in s or '(' + d + ')' in s:
                        if d not in mat_sentences:
                            mat_sentences[d] = [s]
                        else:
                            mat_sentences[d].append(s)

json_str = json.dumps(mat_sentences, indent=4)
with open('mat_sentences.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_str)
'''
tokenizer = BertTokenizer.from_pretrained('mat_bert_cased', do_lower_case=False)
model = BertModel.from_pretrained('mat_bert_cased', output_hidden_states=True).eval()

mat_embeddings = {}
for m in tqdm(list(bran_dict.keys())):
    if m == 'PCE':
        continue
    tmp_sentence = []
    tmp_embedding = []
    for bran in bran_dict[m]:
        if bran in mat_sentences.keys():
            for s in mat_sentences[bran]:
                if bran != m:
                    s = s.replace(' ' + bran + ' ', ' ' + m + ' ')
                    s = s.replace('(' + bran + ')', ' ' + m + ' ')
                    s = s.replace(' ' + m + ' ' + m + ' ', ' ' + m + ' ')
                tmp_sentence.append(s)
    tmp_sentence = list(set(tmp_sentence))
    if len(tmp_sentence) < 4:
        continue
    if len(tmp_sentence) > 100:
        tmp_sentence = sample(tmp_sentence, 100)
    for t in tmp_sentence:
        # 如果m含空格需要特殊处理
        if ' ' in m:
            no_space_m = m.replace(' ', '')
            # print(no_space_m)
            t = t.replace(m, no_space_m)
            embedding = get_embedding(no_space_m, t, tokenizer, model)
        else:
            embedding = get_embedding(m, t, tokenizer, model)
        if embedding is not None:
            tmp_embedding.append(embedding)
    if tmp_embedding:
        mat_embeddings[m] = torch.mean(torch.stack(tmp_embedding), 0)

torch.save(mat_embeddings, 'tensors.pt')
