#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 20:51
# @Author  : LX Ann
# @FileName: simgnn.py
# @Software: PyCharm
# @Email   : 578520080@qq.com


import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch
from extra import feedback_Att, Dense_GCN, Conv_module


"""SimGNN_alx: A neural network approach to fast graph similarity computation"""
class SimGNN(nn.Moudule):

    def __init__(self, args, number_of_labels, device):
        super(SimGNN, self).__init__()
        self.labels = number_of_labels
        self.hist = args.hist
        self.ifDense_GCN = args.ifDense_GCN
        self.feedback = args.feedback
        self.device = device
        self.input_size_of_fc = 32  # 全连接层的输入维度

        if (self.hist == "none"):
            self.input_size_of_fc =  16

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
        features = nn.functional.relu(features) #不能用nn.ReLU()
        features = nn.functional.dropout(features, p=0.3, training=self.training) #不能用nn.dropout(),同时应该注意training的设置

        features = self.conv_2(features, edges)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features, p=0.4, training=self.training) #p表示丢弃的概率

        features = self.conv_3(features, edges)

        return features

    """传入的第二个是转置后的特征矩阵"""
    def calculate_hist(self, embedding_1, embedding_2):
        if (self.hist == "hist"):
            s = torch.mm(embedding_1, embedding_2).detach() #使用detach截断反向传播的梯度流  torch.mm矩阵相乘
            s = s.view(-1, 1)
            hist = torch.histc(s, bins=16) #计算直方图向量，元素被分类到min-max之间等宽的bin中，默认使用数据的最小值和最大值

            return hist

        s = torch.mm(embedding_1, embedding_2)
        ret = self.C(s)
        return ret

    def forward(self, data):
        edge_1 = data["edge_index_1"]
        edge_2 = data["edge_index_2"]
        features_1 = data["features"]
        features_2 = data["features_2"]

        """通过图卷积网络得到每个节点的特征向量"""
        if self.ifDense_GCN is True:
            embedding_1 = self.D_G(edge_1, features_1)
            embedding_2 = self.D_G(edge_2, features_2)

        """节点嵌入生成图嵌入"""
        graph_embedding_1 = self.Att(embedding_1)
        graph_embedding_2 = self.Att(embedding_2)

        """NTN"""
        scores = torch.t(self.NTN(graph_embedding_1, graph_embedding_2)) #图嵌入交互分数

        """Histogram"""
        if (self.hist != "node"):
            h = self.calculate_hist(embedding_1, torch.t(embedding_2))
            scores = torch.cat((scores, h), dim=1).view(1, -1) #score拼接上hist相似向量

        """计算最终的相似分数"""
        ret = self.fc(scores)
        return ret

    def diff(self, abstract_features, edge_index, batch):
        """
        差分池化
        :param abstract_features: 节点特征矩阵
        :param edge_index: 边索引
        :param batch: Batch vector, which assigns each node to a specific example
        :return: pooled_features: Graph feature matrix
        """
        x, mask = to_dense_batch(abstract_features, batch)
        adj = to_dense_adj(edge_index, batch)
        return self.attention(x, adj, mask)

    def consine_attention(self, v1, v2):
        """
        :param v1: (batch, len1, dim)
        :param v2: (batch, len2, dim)
        :return: (batch, len1, len2)
        """
        # (batch, len1, len2)
        a = torch.bmm(v1, v2.permute(0, 2, 1))

        v1_norm = v1.norm(p=2, dim=2, keepdim=True)  #(batch, len1, 1)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1) #(batch, len2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)

    def global_aggregation_info(self, v, agg_func_name):
        """
        :param v: (batch, len, dim)
        :param agg_func_name:
        :return: (batch, len)
        """
        if agg_func_name.lower() == 'max_pool':
            agg_v = torch.max(v, 1)[0]
        elif agg_func_name.lower() == 'fc_max_pool':
            agg_v = self.global_fc_agg(v)
            agg_v = torch.max(agg_v, 1)[0]
        elif agg_func_name.lower() == 'mean_pool':
            agg_v = torch.mean(v, dim=1)
        elif agg_func_name.lower() == 'fc_mean_pool':
            agg_v = self.global_fc_agg(v)
            agg_v = torch.mean(agg_v, dim=1)
        elif agg_func_name.lower() == 'lstm':
            _, (agg_v_last, _) = self.global_lstm_agg(v)
            agg_v = agg_v_last.permute(1, 0, 2).contiguous().view(-1, self.gcn_last_filter * 2)
        else:
            raise NotImplementedError
        return agg_v

    def multi_perspective_match_func(self, v1, v2, w):
        """
        :param v1: (batch, len, dim)
        :param v2: (batch, len, dim)
        :param w: (perspectives, dim)
        :return: (batch, len, perspectives)
        """
        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)        # (1, 1, dim, perspectives)
        v1 = w * torch.stack([v1] * self.perspectives, dim=3)  # (batch, len, dim, perspectives)
        v2 = w * torch.stack([v2] * self.perspectives, dim=3)  # (batch, len, dim, perspectives)
        return functional.consine_simlarity(v1, v2, dim=2)     # (batch, len, perspectives)

    def forward_dense_gcn_layer(self, feat, adj):

        feat_in = feat
        for i in range(1, self.gcn_numbers + 1):
            feat_out = functional.relu(getattr(self, 'gc{}'.format(i))(x=feat_in, adj=adj, mask=None, add_loop=False), inplace=True)
            feat_out = functional.dropout(feat_out, p=self.dropout, training=self.training)
            feat_in = feat_out
        return feat_out


"""可学习的注意模块"""
class Att(nn.Module):

    def __init__(self):
        super(Att, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(16, 16)) #将张量变为可训练的
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding):
        global_context = torch.mean(torch.matmul(embedding, self.weight), dim=0)
        global_context = torch.tanh(global_context)
        att_scores = torch.sigmoid(torch.mm(embedding, global_context.view(-1, 1))) #结果为长为n的得分序列，是列向量
        ret = torch.mm(torch.t(embedding), att_scores)
        return ret

"""神经张量网络，用于对两个向量的相似度进行打分"""
class NTN(nn.Module):

    def __init__(self):
        super(NTN, self).__init__()
        self.W = torch.nn.Parameter(torch.Tensor(16, 16, 16)) #最后一维是K
        self.V = torch.nn.Parameter(torch.Tensor(16, 32))
        self.bias = torch.nn.Parameter(torch.Tensor(16, 1))

        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.V)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2): #两个向量都是列向量
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