#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 20:51
# @Author  : LX Ann
# @FileName: simgnn.py
# @Software: PyCharm
# @Email   : 578520080@qq.com


import torch
import torch.nn as nn
# import torch.nn.functional as functional
from torch_geometric.nn import GCNConv
from utils import feedback_Att, Dense_GCN, Conv_module


"""SimGNN_alx: A neural network approach to fast graph similarity computation"""
class SimGNN(nn.Module):

    def __init__(self, args, number_of_labels, device):

        super(SimGNN, self).__init__()
        self.labels = number_of_labels
        self.hist = args.hist
        self.ifDense_GCN = args.ifDense_GCN
        self.feedback = args.feedback
        self.device = device
        self.input_size_of_fc = 32  # 全连接层的输入维度

        if (self.hist == "none"):
            self.input_size_of_fc = 16

        if self.feedback is True:
            self.Att = feedback_Att()
        else:
            self.Att = Att()

        self.NTN = NTN()

        if (self.ifDense_GCN == True):
            self.D_G = Dense_GCN(self.labels, self.device)
        else:
            self.conv_1 = GCNConv(self.labels, 64)
            self.conv_2 = GCNConv(64, 32)
            self.conv_3 = GCNConv(32, 16)

        if (self.hist == "conv"):
            self.C = Conv_module()

        self.fc = FC(self.input_size_of_fc)

    """三次图卷积"""
    def extract_features(self, edges, features):

        features = self.conv_1(features, edges)
        features = nn.functional.relu(features)  # 注意不能使用nn.ReLU()
        features = nn.functional.dropout(features, p=0.3, training=self.training)  # 不能使用nn.dropout(),同时应该注意training的设置

        features = self.conv_2(features, edges)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features, p=0.4, training=self.training)  # p表示丢弃的概率

        features = self.conv_3(features, edges)

        return features

    """传入的第二个是转置后的特征矩阵"""
    def calculate_hist(self, embedding_1, embedding_2):

        if (self.hist == "hist"):
            s = torch.mm(embedding_1, embedding_2).detach()  # 使用detach,截断反向传播的梯度流   torch.mm 矩阵相乘
            s = s.view(-1, 1)
            hist = torch.histc(s, bins=16)  # 计算直方图向量  元素被分类到min-max之间等宽的bin中；默认使用数据的最小值和最大值，低于min和高于max的元素将被忽略
            hist = hist / torch.sum(hist)  # 归一化  和为1, [0, 1]
            hist = hist.view(1, -1)

            return hist

        s = torch.mm(embedding_1, embedding_2)
        ret = self.C(s)
        return ret

    def forward(self, data):

        edge_1 = data["edge_index_1"]
        edge_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        # print(edge_1.shape)
        # print(features_1.shape)

        """通过图卷积得到 每个节点的特征向量"""
        if self.ifDense_GCN is True:
            embedding_1 = self.D_G(edge_1, features_1)
            embedding_2 = self.D_G(edge_2, features_2)
        else:
            embedding_1 = self.extract_features(edge_1, features_1)
            embedding_2 = self.extract_features(edge_2, features_2)

        # # print(embedding_1)   #  n * 16
        # print("graph1_nodes_num: ", len(embedding_1))
        # print("graph2_nodes_num: ", len(embedding_2))
        # for i in range(len(embedding_1)):
        #     print()
        # # print(embedding_2)
        # print("------")

        ## ---------- 节点嵌入生成图嵌入 ----------
        graph_embedding_1 = self.Att(embedding_1)
        graph_embedding_2 = self.Att(embedding_2)

        ## ---------- NTN ----------
        scores = torch.t(self.NTN(graph_embedding_1, graph_embedding_2)) # 图嵌入交互分数

        #---------- Histogram  ----------
        if (self.hist != "none"):
            h = self.calculate_hist(embedding_1, torch.t(embedding_2))
            scores = torch.cat((scores, h), dim=1).view(1, -1)   # scores拼接上hist相似向量

        # print(scores.shape)
        # 最终的相似分数
        ret = self.fc(scores)
        return ret


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
        att_scores = torch.sigmoid(torch.mm(embedding, global_context.view(-1, 1)))  # 结果为长为n的得分序列,是列向量
        ret = torch.mm(torch.t(embedding), att_scores)
        return ret


"""神经张量网络，用于对两个向量的相似度进行打分"""
class NTN(nn.Module):

    def __init__(self):
        super(NTN, self).__init__()
        self.W = torch.nn.Parameter(torch.Tensor(16, 16, 16))  # 最后一维是K
        self.V = torch.nn.Parameter(torch.Tensor(16, 32))
        self.bias = torch.nn.Parameter(torch.Tensor(16, 1))
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.V)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):  # 注意两个向量都是列向量

        A = torch.mm(torch.t(embedding_1), self.W.view(16, -1))
        A = A.view(16, -1)
        A = torch.mm(torch.t(A), embedding_2)

        B = torch.cat((embedding_1, embedding_2))
        B = torch.mm(self.V, B)

        ret = nn.functional.relu(A + B + self.bias)

        return ret


"""4层全连接网络， input_size to 16 to 8 to 4 to 1"""
class FC(nn.Module):

    def __init__(self, input_size):
        super(FC, self).__init__()
        self.conv_1 = Dense(input_size, 16, act="relu")
        self.conv_2 = Dense(16, 8, act="relu")
        self.conv_3 = Dense(8, 4, act="relu")
        self.conv_4 = Dense(4, 1, act="sigmoid")

    def forward(self, x):
        ret = self.conv_2(self.conv_1(x))
        ret = self.conv_4(self.conv_3(ret))
        return ret


"""封装 预设权重 和 激活函数的Linear"""
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



################################################################
"""
数据集IMDB报错：UnpicklingError: pickle data was truncated
"""