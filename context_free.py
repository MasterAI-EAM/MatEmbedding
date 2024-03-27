# -*- coding: utf-8 -*-
# @Time    : 3/28/2024 1:15 AM
# @Author  : WAN Yuwei
# @FileName: context_free.py
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


def output(outd, model_name, layer_n, pool, cls):
    if 'mat_bert_cased' in model_name:
        short = 'matbert'
    else:
        short = 'bert'
    if cls:
        if not os.path.exists(outd+'context_free_CLS'):
            os.makedirs(outd+'context_free_CLS')
        return f'{outd}context_free_CLS/{short}_{layer_n}_cls.txt'
    else:
        if not os.path.exists(outd+'context_free_name'):
            os.makedirs(outd+'context_free_name')
        return f'{outd}context_free_name/{short}_{layer_n}.txt'


ZT_FILE_PATH = '../for_spearman/zt_ori_84.txt'
# load zt file
df = pd.read_csv(ZT_FILE_PATH, sep='\t', header=None, names=['name', 'value'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", default="C:/Users/Lenovo/Desktop/EnergyBERT/mat_bert_cased", type=str,
                        help="Path of BERT model")
    parser.add_argument("--output_dir", default="../results/", type=str, help="Dir to save output")
    parser.add_argument("--layer_num", default=12, type=int, help="The layer of embedding to extract: [1,12]")
    parser.add_argument("--pool", default=False, type=bool, help="Whether use pool layer")
    parser.add_argument("--cls", default=False, type=bool, help="Whether use cls")
    args = parser.parse_args()

    bert_path = args.bert_path
    output_dir = args.output_dir
    layer_num = args.layer_num
    pool_bool = args.pool
    cls_bool = args.cls

    ext = Extractor(bert_path)
    if cls_bool:
        thermoelectric_embedding = ext.only_CLS("thermoelectric", layer_num)
    else:
        thermoelectric_embedding = ext.avg_all("thermoelectric", layer_num)

    record = {}
    for i, n in enumerate(df['name']):
        record[n] = {}
        if cls_bool:
            name_embedding = ext.only_CLS(n, layer_num)
        else:
            name_embedding = ext.avg_all(n, layer_num)
        similarity = 1 - cosine(name_embedding, thermoelectric_embedding)
        token_num = ext.token_num(n)
        record[n]['embedding'] = name_embedding
        record[n]['sim'] = similarity
        record[n]['token_num'] = token_num

    pred = [item[1]['sim'] for item in record.items()]
    ans = list(df['value'])

    pr, tr = score_to_rank(pred, ans)
    res = stats.spearmanr(pr, tr)
    print('spearman', res)

    embedding_list = [item[1]['embedding'] for item in record.items()]
    anis = calculate_anisotropy(embedding_list)
    print('Anisotropy', anis)

    log = output(output_dir, bert_path, layer_num, pool_bool, cls_bool)
    with open(log, 'w', encoding='utf-8') as f:
        f.write(str(res)+'\n')
        f.write('AnisotropyResult('+str(anis)+')')
