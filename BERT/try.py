# -*- coding: utf-8 -*-
# @Time    : 2022/8/19 17:01
# @Author  : WAN Yuwei
# @FileName: try.py
# @Email: yuweiwan2-c@my.cityu.edu.hk
# @Github: https://github.com/yuweiwan
# @Personal Website: https://yuweiwan.github.io/
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial import distance
from scipy.spatial.distance import cosine
import json
from tqdm import tqdm


def get_embedding(text0, tokenizer, model):
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


tokenizer = BertTokenizer.from_pretrained('mat_bert_cased', do_lower_case=False)
model = BertModel.from_pretrained('mat_bert_cased', output_hidden_states=True).eval()
'''
with open('../pair_list.json', 'r', encoding='utf-8') as f:
    pair_list = json.load(f)
with open('../ori_dict.json', 'r', encoding='utf-8') as f:
    ori_dict = json.load(f)
with open('../mat_dict.json', 'r', encoding='utf-8') as f:
    mat_dict = json.load(f)


# cosine_value = calculate_cosine('contact passivation', '?', tokenizer, model)
center = 'contact passivation'
v1 = get_embedding(center, tokenizer, model)
score_dict = {}
for p in tqdm(pair_list):
    if p['max_abb'] not in score_dict.keys():
        tmp = []
        for f in p['full']:
            v2 = get_embedding(f, tokenizer, model)
            tmp.append(1 - cosine(v1, v2))
        for abb in p['abb']:
            v2 = get_embedding(abb, tokenizer, model)
            tmp.append(1 - cosine(v1, v2))
        score_dict[p['max_abb']] = max(tmp)

for m in tqdm(ori_dict.keys()):
    if m not in score_dict.keys():
        v2 = get_embedding(m, tokenizer, model)
        tmp = [1 - cosine(v1, v2)]
        for o in ori_dict[m]:
            v2 = get_embedding(o, tokenizer, model)
            tmp.append(1 - cosine(v1, v2))
        score_dict[m] = max(tmp)

sorted_score = sorted(score_dict.items(), key=lambda item:item[1], reverse=True)
json_str = json.dumps(sorted_score, indent=4)
with open('sorted_score_berkeley.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_str)
'''
tio2 = get_embedding('gold', tokenizer, model)
ti2o4 = get_embedding('oxygen', tokenizer, model)
print(tio2[:10])
print(ti2o4[:10])
print(distance.cosine(tio2, ti2o4))