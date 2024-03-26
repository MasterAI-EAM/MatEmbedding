# -*- coding: utf-8 -*-
# @Time    : 3/25/2024 3:08 AM
# @Author  : WAN Yuwei
# @FileName: utils.py
# @Email: yuweiwan2-c@my.cityu.edu.hk
# @Github: https://github.com/yuweiwan
# @Personal Website: https://yuweiwan.github.io/
import numpy as np
from scipy.spatial.distance import cosine


def find_token(mat, tok_sen):
    candidate = []
    for i, t in enumerate(tok_sen):
        if mat.startswith(t):
            idx = [i]
            can = t
            tmp = i
            if can == mat:
                candidate.extend(idx)
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


def score_to_rank(prediction, test):
    dict_num = {}
    for i in range(len(test)):
        dict_num[i] = {}
        dict_num[i]['pv'] = prediction[i]
        dict_num[i]['tv'] = test[i]
    sorted_p = sorted(dict_num.items(), key=lambda item: item[1]['pv'], reverse=True)
    sorted_t = sorted(dict_num.items(), key=lambda item: item[1]['tv'], reverse=True)
    count = 1
    for sp in sorted_p:
        dict_num[sp[0]]['pr'] = count
        count += 1
    count = 1
    for sp in sorted_t:
        dict_num[sp[0]]['tr'] = count
        count += 1
    tr = []
    pr = []
    for i in dict_num.keys():
        pr.append(dict_num[i]['pr'])
        tr.append(dict_num[i]['tr'])
    return pr, tr


def calculate_anisotropy(embeddings):
    n = len(embeddings)
    sum_sim = 0
    for i, e in enumerate(embeddings):
        for j, v in enumerate(embeddings):
            if i != j:
                sim = 1 - cosine(e, v)
                sum_sim += sim
    anisotropy = sum_sim / (n * (n - 1))
    return anisotropy

