# -*- coding: utf-8 -*-
# @Time    : 3/29/2024 8:26 AM
# @Author  : WAN Yuwei
# @FileName: context_average.py
# @Email: yuweiwan2-c@my.cityu.edu.hk
# @Github: https://github.com/yuweiwan
# @Personal Website: https://yuweiwan.github.io/
import argparse
from extractor import Extractor
import pandas as pd
from scipy.spatial.distance import cosine
from scipy import stats
from utils import score_to_rank, calculate_anisotropy
import os
import json
from context_free import ZT_FILE_PATH
import random
from tqdm import tqdm
import torch

CENTER_CONTEXT_PATH = '../context_zt/thermo_sens.json'
ZT_CONTEXT_PATH = '../zt_json.json'

random.seed(0)


def get_final_embedding(mat, mode, zt_json, use_alt, layer_idx):
    tmp_embedding = []
    tmp_sentence = []
    for zj in zt_json.keys():
        if mat == zt_json[zj]['ori'] and 'sens' in zt_json[zj].keys():
            tmp_sentence.extend(zt_json[zj]['sens'])
    if len(tmp_sentence) > 100:
        sentence = random.sample(tmp_sentence, 100)
    else:
        sentence = tmp_sentence
    # print(len(sentence), sentence[:5])
    for ts in tqdm(sentence):
        if mode == "cls":
            e_ = ext.only_CLS(ts, layer_num)
        elif mode == "sen":
            e_ = ext.avg_all(ts, layer_num)
        else:
            e_ = ext.avg_need(mat, ts, layer_num)
        if type(e_) != bool:
            tmp_embedding.append(e_)
    if not tmp_embedding:
        if use_alt:
            return ext.avg_all(mat, layer_idx)
        else:
            return None
    else:
        return torch.mean(torch.stack(tmp_embedding), 0)


def output(outd, model_name, layer_n, pool, mode, alt):
    if 'mat_bert_cased' in model_name:
        short = 'matbert'
    else:
        short = 'bert'
    if alt:
        if not os.path.exists(outd + 'context_average_alt'):
            os.makedirs(outd + 'context_average_alt')
        return f'{outd}context_average_alt/{short}_{layer_n}_{mode}.txt'
    else:
        if not os.path.exists(outd + 'context_average_no_alt'):
            os.makedirs(outd + 'context_average_no_alt')
        return f'{outd}context_average_no_alt/{short}_{layer_n}_{mode}.txt'


# load zt file
df = pd.read_csv(ZT_FILE_PATH, sep='\t', header=None, names=['name', 'value'])

with open(ZT_CONTEXT_PATH, 'r', encoding='utf-8') as f:
    zt_context = json.load(f)

with open(CENTER_CONTEXT_PATH, 'r', encoding='utf-8') as f:
    thermo_sen = json.load(f)
thermo_sen = random.sample(thermo_sen, 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", default="C:/Users/Lenovo/Desktop/EnergyBERT/mat_bert_cased", type=str,
                        help="Path of BERT model")
    parser.add_argument("--output_dir", default="../results/", type=str, help="Dir to save output")
    parser.add_argument("--layer_num", default=12, type=int, help="The layer of embedding to extract: [1,12]")
    parser.add_argument("--pool", default=False, type=bool, help="Whether use pool layer")
    parser.add_argument("--mode", default="cls", type=str, help="the mode in extract context-average embeddings: ["
                                                                "sen, need, cls]")
    parser.add_argument("--alt", default=False, type=bool, help="Whether to use alt if no sentence")
    args = parser.parse_args()

    bert_path = args.bert_path
    output_dir = args.output_dir
    layer_num = args.layer_num
    pool_bool = args.pool
    mode_name = args.mode
    alt_bool = args.alt
    print(bert_path, output_dir, layer_num, pool_bool, mode_name, alt_bool)

    ext = Extractor(bert_path)

    tmps = []
    for ts in tqdm(thermo_sen):
        if mode_name == "cls":
            te = ext.only_CLS(ts, layer_num)
        elif mode_name == "sen":
            te = ext.avg_all(ts, layer_num)
        else:
            te = ext.avg_need("thermoelectric", ts, layer_num)
        if type(te) != bool:
            tmps.append(te)
    thermoelectric_embedding = torch.mean(torch.stack(tmps), 0)

    record = {}
    for i, n in enumerate(tqdm(df['name'])):
        name_embedding = get_final_embedding(n, mode_name, zt_context, alt_bool, layer_num)
        if type(name_embedding)==torch.Tensor:
            similarity = 1 - cosine(name_embedding, thermoelectric_embedding)
            print(i, n, similarity)
            token_num = ext.token_num(n)
            record[n] = {}
            record[n]['embedding'] = name_embedding
            record[n]['sim'] = similarity
            record[n]['token_num'] = token_num
            record[n]['value'] = df['value'][i]

    pred = [item[1]['sim'] for item in record.items()]
    ans = [item[1]['value'] for item in record.items()]

    pr, tr = score_to_rank(pred, ans)
    res = stats.spearmanr(pr, tr)
    print('spearman', res)

    embedding_list = [item[1]['embedding'] for item in record.items()]
    anis = calculate_anisotropy(embedding_list)
    print('Anisotropy', anis)

    log = output(output_dir, bert_path, layer_num, pool_bool, mode_name, alt_bool)
    print(log)
    with open(log, 'w', encoding='utf-8') as f:
        f.write(str(res) + '\n')
        f.write('AnisotropyResult(' + str(anis) + ')')
