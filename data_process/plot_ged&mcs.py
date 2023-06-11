#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 10:19
# @Author  : LX Ann
# @FileName: plot_ged&mcs.py
# @Software: PyCharm
# @Email   : 578520080@qq.com


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_metric_distribution(type):
    type_l = ""
    if type == "ged":
        type_l = "ged_astar"
    elif type == "mcs":
        type_l = "mcs_mccreesh2017"

    """读取四个数据集数据"""
    res_path = os.path.abspath(os.path.dirname("__file__")) # data_process
    path = os.path.abspath(os.path.join(res_path, os.path.pardir))
    path = os.path.abspath(os.path.join(path, os.path.pardir)) #

    with open(path + "/datasets" + "/AIDS700nef" + "/" + type + "/aids700nef" + "_" + type_l + "_gidpair_dist_map.pickle", "rb") as handle:
        ged_dict1 = pickle.load(handle)
    tr_x1 = list(ged_dict1.values())  # ged值 mcs值

    with open(path + "/datasets" + "/LINUX" + "/" + type + "/linux" + "_" + type_l + "_gidpair_dist_map.pickle",
              "rb") as handle:
        ged_dict2 = pickle.load(handle)
    tr_x2 = list(ged_dict2.values())  # ged值 mcs值

    with open(path + "/datasets" + "/IMDBMulti" + "/" + type + "/imdbmulti" + "_" + type_l + "_gidpair_dist_map.pickle",
              "rb") as handle:
        ged_dict3 = pickle.load(handle)
    tr_x3 = list(ged_dict3.values())  # ged值 mcs值

    with open(path + "/datasets" + "/PTC" + "/" + type + "/ptc" + "_" + type_l + "_gidpair_dist_map.pickle",
              "rb") as handle:
        ged_dict4 = pickle.load(handle)
    tr_x4 = list(ged_dict4.values())  # ged值 mcs值

    """设置最佳直方图位置"""
    if type == "ged":
        train_bins1 = 25
        train_bins2 = 18
        train_bins3 = 68
        train_bins4 = 74
    else:
        train_bins1 = 11
        train_bins2 = 7
        train_bins3 = 50
        train_bins4 = 64

    train_avg1 = np.mean(tr_x1)
    train_avg2 = np.mean(tr_x2)
    train_avg3 = np.mean(tr_x3)
    train_avg4 = np.mean(tr_x4)

    plt.style.use('seaborn-whitegrid')
    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 10,
    }
    plt.rc('font', family='Times New Roman')


    plt.subplot(2, 2, 1)
    plt.style.use('seaborn-whitegrid')
    if type == "ged":
        plt.xlabel('GED', font1)
        plt.xlim(xmax =20)
    else:
        plt.xlabel('MCS', font1)
        plt.xlim(xmax=10)
    plt.ylabel('Count', font1)
    plt.yticks(fontproperties='Times New Roman', size=11)
    plt.xticks(fontproperties='Times New Roman', size=11)
    plt.hist(tr_x1, bins=train_bins1, color="darkcyan", lw=10)
    # y = norm.pdf(train_bins1, np.mean(tr_x1), 15)
    plt.title('Graph Pair - ' + str(len(tr_x1)) + ' (AIDS)', font1)  # 图对数量
    plt.axvline(train_avg1, color='k', linestyle='dashed', linewidth=1.2)
    plt.ticklabel_format(style='sci', scilimits=(0, 0))
    min_ylim, max_ylim = plt.ylim()
    plt.text(train_avg1 + 0.6, max_ylim * 0.9, '  Mean: {:.2f}'.format(train_avg1), font1)

    plt.rc('font', family='Times New Roman')
    plt.subplot(2, 2, 2)
    plt.style.use('seaborn-whitegrid')
    if type == "ged":
        plt.xlabel('GED', font1)
        plt.xlim(xmax = 15)
    else:
        plt.xlabel('MCS', font1)
        plt.xlim(xmax=10)
    plt.ylabel('Count', font1)
    plt.yticks(fontproperties='Times New Roman', size=11)
    plt.xticks(fontproperties='Times New Roman', size=11)
    plt.hist(tr_x2, bins=train_bins2, color="darkcyan", lw=10)
    plt.title('Graph Pair - ' + str(len(tr_x2)) + ' (LINUX)', font1)  # 图对数量
    plt.axvline(train_avg2, color='k', linestyle='dashed', linewidth=1.2)
    plt.ticklabel_format(style='sci', scilimits=(0, 0))
    min_ylim, max_ylim = plt.ylim()
    plt.text(train_avg2 + 0.3, max_ylim * 0.9, ' Mean: {:.2f}'.format(train_avg2), font1)

    plt.rc('font', family='Times New Roman')
    plt.subplot(2, 2, 3)
    plt.style.use('seaborn-whitegrid')
    if type == "ged":
        plt.xlabel('GED', font1)
        plt.xlim(xmax = 1000)
    else:
        plt.xlabel('MCS', font1)
        plt.xlim(xmax=30)
    plt.ylabel('Count', font1)
    plt.yticks(fontproperties='Times New Roman', size=11)
    plt.xticks(fontproperties='Times New Roman', size=11)
    plt.hist(tr_x3, bins=train_bins3, color="darkcyan", lw=10)
    plt.title('Graph Pair - ' + str(len(tr_x3)) + ' (IMDB)', font1)  # 图对数量
    plt.axvline(train_avg3, color='k', linestyle='dashed', linewidth=1.2)
    plt.ticklabel_format(style='sci', scilimits=(0, 0))
    min_ylim, max_ylim = plt.ylim()
    plt.text(train_avg3, max_ylim * 0.9, ' Mean: {:.2f}'.format(train_avg3), font1)

    plt.rc('font', family='Times New Roman')
    plt.subplot(2, 2, 4)
    plt.style.use('seaborn-whitegrid')
    if type == "ged":
        plt.xlabel('GED', font1)
        plt.xlim(xmax = 200)
    else:
        plt.xlabel('MCS', font1)
        plt.xlim(xmax=40)
    plt.ylabel('Count', font1)
    plt.yticks(fontproperties='Times New Roman', size=11)
    plt.xticks(fontproperties='Times New Roman', size=11)
    plt.hist(tr_x4, bins=train_bins4, color="darkcyan", lw=10)
    plt.title('Graph Pair - ' + str(len(tr_x4)) + ' (PTC)', font1)  # 图对数量
    plt.axvline(train_avg4, color='k', linestyle='dashed', linewidth=1.2)
    plt.ticklabel_format(style='sci', scilimits=(0, 0))
    min_ylim, max_ylim = plt.ylim()
    plt.text(train_avg4 + 2.3, max_ylim * 0.9, ' Mean: {:.2f}'.format(train_avg4), font1)


    plt.tight_layout()  # 调整整体空白
    # plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
    plt.show()


if __name__ == '__main__':
    """运行时设置type"""
    type = "ged"
    # type = "mcs"

    plot_metric_distribution(type)



