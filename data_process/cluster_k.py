#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 12:26
# @Author  : LX Ann
# @FileName: cluster_k.py
# @Software: PyCharm
# @Email   : 578520080@qq.com


import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from glob import glob
from os.path import basename
from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN


path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))  # /src
labeled = True #表明数据是否带有标记

def load_graph(dir):
    graphs = []
    for file in glob(dir + "/*.gexf"):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph["gid"] = gid
        graphs.append(g)
        if not nx.is_connected(g):  #判断连通性
            raise RuntimeError('{} not connected'.format(gid))
    return graphs


def read_dataset_graph(dataset):
    dir_train = path + "/datasets/" + dataset + "/train"
    dir_test = path + "/datasets/" + dataset + "/test"
    train_graphs = load_graph(dir_train)
    test_graphs = load_graph(dir_test)
    total_graphs = train_graphs + test_graphs
    m = len(train_graphs)
    n = len(test_graphs)
    # # print(m) # 560
    # # print(n) # 560
    # print("total graphs: ", m + n)
    return m + n, total_graphs

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
        for node in g.nodes().data(): #形如 ('7', {'type': 'C', 'label': '7'})
            labels[new_id[node[0]]] = int(node[1]["label"]) + 1

    edges = []
    for edge in g.edges().data(): #形如 ('7', '4', {'valence': 2, 'id': '6'})
        edges.append([new_id[edge[0]], new_id[edge[1]]])

    return edges, labels


def structure(total, total_graphs):
    graph_cluster_structure = []

    clique_len_list = []  # 数据集train图的clique规模分布
    _1_core_len_list = []
    _2_core_len_list = []
    _3_core_len_list = []

    _1_2_core_len_list = []
    _3_29_core_len_list = []

    star_len_list = []
    for i in range(total):
        edges, labels = get_graph_message(total_graphs[i])
        # print("edges: ", edges)

        G = nx.Graph()
        g_edges = edges
        G.add_edges_from(g_edges)
        # degree = nx.degree_histogram(G)

        graph_structure = []

        # nx.draw(G, with_labels=True)
        #
        res_clique = nx.find_cliques(G)
        # res_1_core = nx.k_core(G, 1)
        # res_2_core = nx.k_core(G, 2)
        # res_3_core = nx.k_core(G, 3)
        # res_star = find_star(G)

        # print(res_1_core)

        # plt.subplot(1, 2, 1)
        # nx.draw(G, with_labels=True)

        # print("len_1_core: ",  len(res_1_core))
        # plt.subplot(1, 2, 2)
        # nx.draw(res_1_core, with_labels=True)

        # 求clique list长度
        j = 0
        for item in res_clique:
            clique_len_list.append(len(item))
            # j += 1

        # for j in range(len(res_star)):
        #     star_len_list.append(res_star[j])


        clique_size_mean = np.mean(clique_len_list)
        graph_structure.append(clique_size_mean)

        # star_size_mean = np.mean(star_len_list)
        # graph_structure.append(star_size_mean)

        # _1_core_len_list.append(len(res_1_core))
        # _2_core_len_list.append(len(res_2_core))
        # _3_core_len_list.append(len(res_3_core))

        # core_size_mean = (1 * len(res_1_core) + 2 * len(res_2_core) ) / (len(res_1_core) + len(res_2_core))
        # graph_structure.append(core_size_mean)

        # print(len(res_1_core))
        # print(len(res_2_core))
        print(clique_size_mean)

        # print(graph_structure)
        graph_cluster_structure.append(graph_structure)

    return graph_cluster_structure


if __name__ == '__main__':

    aids_total, aids_total_graphs = read_dataset_graph("AIDS700nef")
    aids_structure = structure(aids_total, aids_total_graphs)

    linux_total, linux_total_graphs = read_dataset_graph("LINUX")
    linux_structure = structure(linux_total, linux_total_graphs)

    ptc_total, ptc_total_graphs = read_dataset_graph("PTC")
    ptc_structure = structure(ptc_total, ptc_total_graphs)

    # imdb_total, imdb_total_graphs = read_dataset_graph("IMDBMulti")
    # imdb_structure = structure_imdb(imdb_total, imdb_total_graphs)

    # '利用SSE选择k'
    SSE0 = []  # 存放每次结果的误差平方和
    for k in range(1, 9):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(aids_structure)
        if k == 2:
            res0Series = pd.Series(estimator.labels_)
            # print(estimator.labels_)
            res0 = res0Series[res0Series.values == 0]
            res1 = res0Series[res0Series.values == 1]
            print("aids各类别的数据\n", res0.size, " ", res1.size)

        SSE0.append(estimator.inertia_)

    SSE2 = []
    for k in range(1, 9):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(linux_structure)
        if k == 2:
            res0Series = pd.Series(estimator.labels_)
            # print(estimator.labels_)
            res0 = res0Series[res0Series.values == 0]
            res1 = res0Series[res0Series.values == 1]
            print("linux各类别的数据\n", res0.size, " ", res1.size)
        SSE2.append(estimator.inertia_)

    SSE3 = []
    for k in range(1, 9):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(ptc_structure)
        if k == 2:
            res0Series = pd.Series(estimator.labels_)
            # print(estimator.labels_)
            res0 = res0Series[res0Series.values == 0]
            res1 = res0Series[res0Series.values == 1]
            print("ptc各类别的数据\n", res0.size, " ", res1.size)
        SSE3.append(estimator.inertia_)

    # SSE1 = []
    # for k in range(1, 9):
    #     estimator = KMeans(n_clusters=k)  # 构造聚类器
    #     estimator.fit(imdb_structure)
    #     if k == 3:
    #         res0Series = pd.Series(estimator.labels_)
    #         # print(estimator.labels_)
    #         res0 = res0Series[res0Series.values == 0]
    #         res1 = res0Series[res0Series.values == 1]
    #         res2 = res0Series[res0Series.values == 2]
    #         print("imdb各类别的数据\n", res0.size, " ", res1.size, " ", res2.size)
    #     SSE1.append(estimator.inertia_)

    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 12,
    }
    plt.rc('font', family='Times New Roman')

    X = range(1, 9)
    plt.xlabel('Number of clusters K', font1)
    plt.ylabel('Inertia', font1)

    plt.title("Inertia of k-Means versus number of clusters")
    plt.plot(X,SSE0,'*-', color="c", markersize=10)
    plt.plot(X,SSE2,'x-', color='#F1404B', markersize=8)
    plt.plot(X,SSE3,'^-', color='#00A2DE',markersize=8)
    # plt.plot(X, SSE1, 'o-', color='#F0DD92', markersize=8)
    # plt.legend(['IMDB'], loc='upper right')
    plt.legend(['AIDS', 'LINUX', 'PTC'], loc='upper right')
    plt.grid(ls='-.')  # 绘制背景线
    plt.ticklabel_format(style='sci', scilimits=(0, 0))
    plt.tight_layout()
    plt.show()

    # plt.plot(x3, y3, label='IMDB', color='#F0DD92')
    # plt.plot(x4, y4, label='PTC', color='#00A2DE')
    # plt.plot(x2, y2, label='LINUX', color='#F1404B')
    # plt.plot(x, y, label='AIDS', color="c")

    X = range(1, 9)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title("Inertia of k-Means versus number of clusters")
    plt.subplot(121)
    # plt.plot(X,SSE1,'o-')
    # plt.legend(['IMDBMulti'], loc='upper right')
    # plt.plot(X,SSE1,'o-')
    # plt.legend(['AIDS700nef'], loc='upper right')
    plt.grid(ls='-.')  # 绘制背景线
    plt.tight_layout()
    plt.show()







# imdb_structure
# aids_structure
# ptc_structure
# linux_structure

# graph_structure_cluster = np.array(imdb_structure)
# y_pred = DBSCAN().fit_predict(graph_structure_cluster)
# plt.subplot(121)
# plt.scatter(graph_structure_cluster[:, 0], graph_structure_cluster[:, 1],  c=y_pred)
#
#
# graph_structure_cluster = np.array(imdb_structure)
# # print(graph_structure_cluster)
# estimator = KMeans(n_clusters=2) #构造聚类器   max_iter=100,
# estimator.fit(graph_structure_cluster)#聚类
# y = estimator.predict(graph_structure_cluster)
# centers = estimator.cluster_centers_
# # print(y)
# plt.subplot(122)
# plt.scatter(graph_structure_cluster[:, 0], graph_structure_cluster[:, 1], s=3, c=y)
# plt.scatter(centers[:, 0], centers[:, 1], s=10, c='r')
# plt.show()