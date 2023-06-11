#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 21:42
# @Author  : LX Ann
# @FileName: plot_loss.py
# @Software: PyCharm
# @Email   : 578520080@qq.com

"""
plot loss
"""


import csv
import matplotlib.pyplot as plt
import numpy as np
import os


bool = True

# if args == '--train':
#     bool = True
# elif args =='--test':
#     bool = False
# else:
#     bool = True

"""先数据处理，拿到log.txt中的前两列数据"""
def dataprocess():
    # read data
    data = []
    res_path = os.path.abspath(os.path.dirname("__file__"))   # /src
    # 修改模型结果的地址
    # res_path = res_path + "/GEDLogs/aids700nef@2022-05-01@16-43-32/log_aids.txt"
    # res_path = res_path + "/GEDLogs/linux@2022-05-02@21-15-31/log_aids.txt"
    res_path = res_path + "/GEDLogs/ptc@2022-05-02@16-08-39/log_aids.txt"
    # res_path = res_path + "/GEDLogs/log_linux_1.txt"

    with open(res_path, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1].split(','))
    data = np.array(data)
    print(data.shape)

    # data process
    data2 = []
    for index, row in enumerate(data.T):
        if index < len(data.T)-1:
            data2.append(row)
    data2 = np.array(data2)
    # print(data2)                    # 拿到了前2列数据

    # write
    a = data2.T
    with open(res_path, 'w') as f:
        for i in range(len(a)):
            for j in range(0, 2):
                if j == 0:
                    f.write(str(a[i][j]) + ',')
                else:
                    f.write(str(a[i][j]) + '\n')

"""处理一次后就不用调用"""
# dataprocess()


def plot_loss(res_path1, res_path2, res_path3, res_path4):

    x = []
    y = []
    with open(res_path1, 'r') as error_graph1:
        plots = csv.reader(error_graph1, delimiter=',')
        for row in plots:
            x.append(int(row[0]))
            y.append(float(row[1]) / 1000)
    x2 = []
    y2 = []
    with open(res_path2, 'r') as error_graph2:
        plots = csv.reader(error_graph2, delimiter=',')
        for row in plots:
            x2.append(int(row[0]))
            y2.append(float(row[1]) / 1000)
        # title = 'AIDS'  #PTC 94600

    x3 = []
    y3 = []
    with open(res_path3, 'r') as error_graph3:
        plots = csv.reader(error_graph3, delimiter=',')
        for row in plots:
            x3.append(int(row[0]))
            y3.append(float(row[1]) / 3200)
        # title = 'AIDS'  #PTC 94600
    x4 = []
    y4 = []
    with open(res_path4, 'r') as error_graph4:
        plots = csv.reader(error_graph4, delimiter=',')
        for row in plots:
            x4.append(int(row[0]))
            y4.append(float(row[1]) / 1800)
        # title = 'AIDS'  #PTC 94600

    plt.rc('font', family='Times New Roman')
    plt.plot(x3, y3, label='IMDB', color='#F0DD92')
    plt.plot(x4, y4, label='PTC', color='#00A2DE')
    plt.plot(x2, y2, label='LINUX', color='#F1404B')
    plt.plot(x, y, label='AIDS', color="c")

    plt.ylabel('Error Rate', size=12)
    plt.xlabel('Epochs', size=12)

    # plt.yticks(np.arange(0.00, 0.10, step=0.02))
    # plt.ylim(ymin=-0.005)

    # plt.title(title)
    plt.legend()
    plt.grid(ls='-.')  # 绘制背景线
    plt.tight_layout()
    plt.show()

    # plt.savefig("../../saved/loss.png")


if __name__ == '__main__':

    """修改这里路径"""
    res_path = os.path.abspath(os.path.dirname("__file__"))  # /data_process

    """ged"""
    res_path1 = res_path + "/GEDLogs/loss/log_aids.txt"
    res_path2 = res_path + "/GEDLogs/loss/log_linux.txt"
    res_path3 = res_path + "/GEDLogs/loss/log_imdb.txt"
    res_path4 = res_path + "/GEDLogs/loss/log_ptc.txt"

    # """mcs"""
    # res_path1 = res_path + "/GEDLogs/loss/log_aids_1.txt"
    # res_path2 = res_path + "/GEDLogs/loss/log_linux_1.txt"
    # res_path3 = res_path + "/GEDLogs/loss/log_imdb_1.txt"
    # res_path4 = res_path + "/GEDLogs/loss/log_ptc_1.txt"

    plot_loss(res_path1, res_path2, res_path3, res_path4)

