# encoding: utf-8
# @email: enoche.chow@gmail.com
"""
############################
"""

from logging import getLogger

import numpy as np


def recall_at_k(r, k, all_poss_num):
    r = list(r)
    r = np.asfarray(r)[:k]
    return len(r) / all_poss_num


def hit_at_k(r, k):
    r = list(r)
    r = np.asfarray(r, dtype=np.int)[:k]
    r = r[r>0]
    if len(r) > 0:
        return 1.
    else:
        return 0.



def recall_(pos_index, pos_len):
    # print(pos_len.shape, pos_index.shape)
    # exit()
    recall = []
    for topk in range(0, pos_index.shape[1]):
        temp_reack = 0.0
        for i, user_topk_res in enumerate(pos_index):
            temp_res = np.array(user_topk_res, dtype=np.int)
            temp_reack += sum(temp_res[:topk]) / pos_len[i]
        recall.append(temp_reack/pos_index.shape[0])

    return recall

def hk_(pos_index, pos_len):
    # Recall: average single users recall ratio.
    hit = np.array([hit_at_k(line, 20) for line in pos_index]).sum()
    return [hit / len(pos_len) for i in range(20)]


def recall2_(pos_index, pos_len):
    r"""
    All hits are summed up and then averaged for recall.
    :param pos_index:
    :param pos_len:
    :return:
    """
    rec_cum = np.cumsum(pos_index, axis=1)
    rec_ret = rec_cum.sum(axis=0) / pos_len.sum()
    return rec_ret


def ndcg_(pos_index, pos_len):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality.
    Through normalizing the score, users and their recommendation list results in the whole test set can be evaluated.
    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in u^{te}NDCG_u@K}}{|u^{te}|}
        \end{gather}

    :math:`K` stands for recommending :math:`K` items.
    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.
    :math:`2^{rel_i}` equals to 1 if the item hits otherwise 0.
    :math:`U^{te}` is for all users in the test set.
    """
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result.mean(axis=0)

def ndcg2_(pos_index, pos_len):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality.
    Through normalizing the score, users and their recommendation list results in the whole test set can be evaluated.
    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in u^{te}NDCG_u@K}}{|u^{te}|}
        \end{gather}

    :math:`K` stands for recommending :math:`K` items.
    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.
    :math:`2^{rel_i}` equals to 1 if the item hits otherwise 0.
    :math:`U^{te}` is for all users in the test set.
    """
    ndcg = [0.0 for topk in range(0,pos_index.shape[1])]
    for topk in range(0,pos_index.shape[1]):
        for user_topk_res in pos_index:
            temp_res = np.array(user_topk_res[:topk], dtype=np.int)
            num_hit = sum(temp_res)
            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_hit, topk)):
                max_ndcg_score += 1 / (np.log(i + 2))
            if max_ndcg_score == 0:
                continue

            for i, item_flag in enumerate(user_topk_res):
                if item_flag:
                    ndcg_score += 1 / (np.log(i + 2))

            ndcg[topk] += ndcg_score / max_ndcg_score

    ndcg = np.array(ndcg)
    return np.divide(ndcg, pos_index.shape[0])


def map_(pos_index, pos_len):
    r"""MAP_ (also known as Mean Average Precision) The MAP is meant to calculate Avg. Precision for the relevant items.
    Note:
        In this case the normalization factor used is :math:`\frac{1}{\min (m,N)}`, which prevents your AP score from
        being unfairly suppressed when your number of recommendations couldn't possibly capture all the correct ones.

    .. _map: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms

    .. math::
        \begin{align*}
        \mathrm{AP@N} &= \frac{1}{\mathrm{min}(m,N)}\sum_{k=1}^N P(k) \cdot rel(k) \\
        \mathrm{MAP@N}& = \frac{1}{|U|}\sum_{u=1}^{|U|}(\mathrm{AP@N})_u
        \end{align*}
    """
    pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    sum_pre = np.cumsum(pre * pos_index.astype(np.float), axis=1)
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
    result = np.zeros_like(pos_index, dtype=np.float)
    for row, lens in enumerate(actual_len):
        ranges = np.arange(1, pos_index.shape[1]+1)
        ranges[lens:] = ranges[lens - 1]
        result[row] = sum_pre[row] / ranges
    return result.mean(axis=0)


def precision_(pos_index, pos_len):
    r"""Precision_ (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances
    .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    .. math::
        \mathrm {Precision@K} = \frac{|Rel_u \cap Rec_u|}{Rec_u}

    :math:`Rel_u` is the set of items relavent to user :math:`U`,
    :math:`Rec_u` is the top K items recommended to users.
    We obtain the result by calculating the average :math:`Precision@K` of each user.
    """
    rec_ret = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    return rec_ret.mean(axis=0)


"""Function name and function mapper.
Useful when we have to serialize evaluation metric names
and call the functions based on deserialized names
"""
metrics_dict = {
    'ndcg': ndcg2_,
    # 'ndcg2': ndcg2_,
    'recall': recall2_,
    # 'recall2': recall2_,
    'precision': precision_,
    'map': map_,
    'hk':hk_,
}
