#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 20:31
# @Author  : LX Ann
# @FileName: graphpairs_compute.py
# @Software: PyCharm
# @Email   : 578520080@qq.com

import os
import pickle
import networkx as nx
from glob import glob
from os.path import basename

current_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))  #源代码所在的路径   /src
path = os.path.abspath(os.path.join(current_path, os.pardir))
labeled = True  # 表明数据是否带有标记

def load_graph(dir):
    # print(dir)
    graphs = []
    for file in glob(dir + "/*.gexf"):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph["gid"] = gid
        graphs.append(g)
        if not nx.is_connected(g):  #判断连通性
            raise RuntimeError('{} not connected'.format(gid))

    return graphs

def load_data(dataset):  #eg: datasets = "AIDS700nef"
    tmp_path = path + "/datasets" + "/" + dataset

    train_graphs = []
    test_graphs = []

    train_path = tmp_path + "/train"
    test_path = tmp_path + "/test"

    train_graphs = load_graph(train_path)
    test_graphs = load_graph(test_path)

    return train_graphs, test_graphs

def get_graph_message(g):
    """
    输入一张图，返回 edges 和 labels
    """
    v = len(g.nodes())
    labels = [1 for i in range(v)]

    """重新编号"""
    id = 0
    new_id = {}
    for node in g.nodes():
        new_id[node] = id
        id += 1

    if labeled == True:
        hsh = list(g.nodes())
        for node in g.nodes().data():  #形如 ('7', {'type': 'C', 'label': '7'})
            labels[new_id[node[0]]] = int(node[1]["label"]) + 1

    edges = []
    for edge in g.edges().data():      #形如 ('7', '4', {'valence': 2, 'id': '6'})
        edges.append([new_id[edge[0]], new_id[edge[1]]])

    return edges, labels


def make_graph_pair(g_1, g_2, ged_dict):
    """
    输入两张图，输出一个dict
    """
    graph_1, labels_1 = get_graph_message(g_1)
    graph_2, labels_2 = get_graph_message(g_2)
    ged = ged_dict[(g_1.graph["gid"], g_2.graph["gid"])]

    ret = {}
    ret["graph_1"] = graph_1
    ret["graph_2"] = graph_2
    ret["labels_1"] = labels_1
    ret["labels_2"] = labels_2
    ret["ged"] = ged

    return ret


def gp_compute(dataset, type):
    type_l = ""
    if type == "ged":
        type_l = "ged_astar"
    else:
        type_l = "mcs_mccreesh2017"

    train_g, test_g = load_data(dataset)
    print("train graphs: ", len(train_g))
    print("test graphs: ", len(test_g))

    pre = ""
    if dataset == "AIDS700nef":
        pre = "aids700nef"
    elif dataset == "IMDBMulti":
        pre = "imdbmulti"
    elif dataset == "PTC" or dataset == "Test_PTC":
        pre = "ptc"
    elif dataset == "LINUX":
        pre = "linux"

    with open(path + "/datasets" + "/" + dataset + "/" + type + "/" + pre + "_" + type_l + "_gidpair_dist_map.pickle", "rb") as handle:
        ged_dict = pickle.load(handle)
    print("graph_pair_ged_dict: ", len(ged_dict))  # 94600 = 275 * 275 + 275 * 69  = 275 * 344    即训练集图数 * 总图数

    train_pair = []
    test_pair = []
    n = len(train_g)  # 275   220 : 55
    m = len(test_g)  # 69

    for i in range(n):
        for j in range(i, n, 1):
            train_pair.append(make_graph_pair(train_g[i], train_g[j], ged_dict))  # train_graph - train_graph
        for j in range(m):
            test_pair.append(make_graph_pair(test_g[j], train_g[i], ged_dict))  # train_graph - test_graph

    print("train_pair: ", len(train_pair))  # 275个训练图的组合             1 + 2 + ... + 275 = 275 * 276 / 2 = 37950
    print("test_pair: ", len(test_pair))  # 275个训练图与69个测试图的组合   275 * 69 = 18975

    with open(path + "/datasets/" + dataset + "/" + type + "/train_data.pickle", "wb") as handle:
        pickle.dump(train_pair, handle)
    with open(path + "/datasets/" + dataset + "/" + type + "/test_data.pickle", "wb") as handle:
        pickle.dump(test_pair, handle)


if __name__ == '__main__':

    """修改这里数据集"""
    dataset = "AIDS700nef"     # "AIDS700nef" / "IMDBMulti" / "PTC" / "LINUX" / "Test_PTC"
    """ged预测 or mcs预测任务"""
    type = "ged"               # "ged" / "mcs"

    gp_compute(dataset, type)  # 计算图对


