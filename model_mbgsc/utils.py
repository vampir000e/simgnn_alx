#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 15:34
# @Author  : LX Ann
# @FileName: utils.py
# @Software: PyCharm
# @Email   : 578520080@qq.com

import os
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from texttable import Texttable
from scipy import stats

"""带有err_feedback机制的Att"""
class feedback_Att(nn.Module):

    def __init__(self):

        super(feedback_Att, self).__init__()
        self.att1 = Att()
        self.conv = Conv1d(1, 10, kernel_size=9, padding=4)
        self.att2 = Att()

    def forward(self, embedding):

        global_e_1 = self.att1(embedding)
        de_embedding = self.conv(global_e_1.view(1, 1, 16)).view(10, 16)
        if embedding.shape[0] < de_embedding.shape[0]:
            de_embedding = de_embedding[0:embedding.shape[0]]
        else:
            embedding = embedding[0:de_embedding.shape[0]]
        res = de_embedding - embedding
        global_e_2 = self.att2(res)

        return global_e_1 + global_e_2


"""可学习的注意模块"""
class Att(nn.Module):

    def __init__(self):
        super(Att, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(16, 16))  # 将张量变为可训练的
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding):
        # mean = torch.mean(embedding, dim = 0, keep_dim = True) #需要划分batch,但是使用for循环使得一次只有一张图通过该模块，故dim=0
        # global_context = torch.tanh(torch.mm(mean, self.weight))
        global_context = torch.mean(torch.matmul(embedding, self.weight), dim=0)
        global_context = torch.tanh(global_context)

        # print(embedding.shape) # torch.Size([16])
        # print(global_context.shape) # torch.Size([*, 16])

        att_scores = torch.sigmoid(torch.mm(embedding, global_context.view(-1, 1)))  # 结果为长为n的得分序列,是列向量
        ret = torch.mm(torch.t(embedding), att_scores)
        return ret


"""封装 预设 权重 和激活函数的 nn.Conv1d"""
class Conv1d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True, act="relu"):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)

        self.act = nn.ReLU()
        if (act == "lrelu"):
            self.act = nn.LeakyReLU(0.2)
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv.bias.data.zero_()

    def forward(self, x):
        return self.act(self.conv(x))


class Dense_GCN(nn.Module):

    def __init__(self, number_of_labels, device):
        super(Dense_GCN, self).__init__()

        self.labels = number_of_labels
        self.conv_1 = GCNConv(self.labels, 64)
        self.conv_2 = GCNConv(64, 32)
        self.conv_3 = GCNConv(32, 16)
        self.fc = Dense(64+32+16, 16, act="relu")
        self.device = device

    def forward(self, edges, features):
        edges = edges.to(self.device)
        features = features.to(self.device)

        features = self.conv_1(features, edges)
        features = nn.functional.relu(features)
        features1 = nn.functional.dropout(features, p=0.3, training=self.training)

        features2 = self.conv_2(features1, edges)
        features2 = nn.functional.relu(features2)
        features2 = nn.functional.dropout(features2, p=0.4, training=self.training) # p丢弃率

        features3 = self.conv_3(features2, edges)
        features3 = nn.functional.relu(features3)

        ret = torch.cat((features1, features2, features3), 1)
        ret = self.fc(ret)

        return ret


"""用多个卷积层来对内积矩阵进行打分"""
class Conv_module(nn.Module):

    def __init__(self, act="relu"):
        super(Conv_module, self).__init__()
        self.conv1 = Conv(1, 8, act=act)
        self.conv2 = Conv(8, 32, act=act)
        self.conv3 = Conv(32, 16, act=act)
        self.act = nn.Sigmoid()

    def forwrd(self, x):
        tmp = x.view(-1, 1, x.shape[0], x.shape[1])

        ret = self.conv1(tmp)
        ret = self.conv2(ret)
        ret = self.conv3(ret)

        tmp = torch.mean(ret, dim=2)
        ttmp = torch.mean(tmp, dim=2)

        return self.act(ttmp)

"""封装， 预设权重和激活函数的Linear"""
class Dense(nn.Module):

    def __init__(self, in_size, out_size, bias=True, act="relu"):
        super(Dense, self).__init__()
        self.conv = nn.Linear(in_size, out_size, bias)
        self.act = nn.ReLU()
        if (act == "sigmoid"):
            self.act = nn.Sigmoid()

        """预设权重"""
        nn.init.kaiming_normal_(self.conv.weight)
        if bias is True:
            self.conv.bias.data.zero_()

    def forward(self, x):
        return self.act(self.conv(x))


"""封装 预设权重和激活函数的nn.Conv1d"""
class Conv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True, act="relu"):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.act = nn.ReLU()

        if (act == "lrrelu"):
            self.act = nn.LeakyReLU(0.2)
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv.bias.data.zero_()

    def forward(self, x):
        return self.act(self.conv(x))


def tab_printer(args):
    """
    参数设置打印
    :param args:
    :return:
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows(
        [["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    )
    print(t.draw())


def random_id(graph, label):
    tmp_graph = []
    for edge in graph:
        tmp_graph.append(edge.copy())

    n = np.shape(label)[0]
    iid = [i for i in range(n)]
    tmp_label = [0 for i in range(n)]

    np.random.shuffle(iid)  #换称号,i的称号换成iid[i]

    for edge in tmp_graph:
        edge[0], edge[1] = iid[edge[0]], iid[edge[1]]
    for i in range(n):
        tmp_label[iid[i]] = label[i]

    return tmp_graph, tmp_label

def write_log_file(file_name_path, log_str, print_flag=True):
    if print_flag:
        print(log_str)
    if log_str is None:
        log_str = 'None'
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'a+') as log_file:
            log_file.write(log_str + '\n')
    else:
        with open(file_name_path, 'w+') as log_file:
            log_file.write(log_str + '\n')

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 'Make dirs of # {} '.format(directory)
    else:
        return "the dirs already exist! Cannot be created"


def metrics_spearmanr_rho(true, predication):
    assert true.shape == predication.shape
    rho, p_val = stats.spearmanr(true, predication)
    return rho


def metrics_kendall_tau(true, predication):
    assert true.shape == predication.shape
    tau, p_val = stats.kendalltau(true, predication)
    return tau


def metrics_mean_square_error(true, predication):
    assert true.shape == predication.shape
    mse = (np.square(true - predication).mean())
    return mse

def computing_precision_ks(trues, predictions, ks, inclusive=True, rm=0):
    assert trues.shape == predictions.shape
    m, n = trues.shape

    precision_ks = np.zeros((m, len(ks)))
    inclusive_final_true_ks = np.zeros((m, len(ks)))
    inclusive_final_pred_ks = np.zeros((m, len(ks)))

    for i in range(m):
        for k_idx, k in enumerate(ks):
            assert (type(k) is int and 0 < k < n)
            true_ids, true_k = top_k_ids(trues, i, k, inclusive, rm)
            pred_ids, pred_k = top_k_ids(predictions, i, k, inclusive, rm)
            precision_ks[i][k_idx] = min(len(set(true_ids).intersection(set(pred_ids))), k) / k
            inclusive_final_true_ks[i][k_idx] = true_k
            inclusive_final_pred_ks[i][k_idx] = pred_k
    return np.mean(precision_ks, axis=0), np.mean(inclusive_final_true_ks, axis=0), np.mean(inclusive_final_pred_ks, axis=0)

def top_k_ids(query, qid, k, inclusive, rm=0):
    sort_id_mat = np.argsort(query, kind='mergesort')[:, ::-1]
    _, n = sort_id_mat.shape
    if k < 0 or k >= n:
        raise RuntimeError('Invalid k {}'.format(k))
    if not inclusive:
        return sort_id_mat[qid][:k]

    while k < n:
        cid = sort_id_mat[qid][k - 1]
        nid = sort_id_mat[qid][k]
        if abs(query[qid][cid] - query[qid][nid]) <= rm:
            k += 1
        else:
            break
    return sort_id_mat[qid][:k], k

