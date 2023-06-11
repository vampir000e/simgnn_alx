#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 12:49
# @Author  : LX Ann
# @FileName: plot.py
# @Software: PyCharm
# @Email   : 578520080@qq.com


################ Layer ####################
# import matplotlib.pyplot as plt
# import numpy as np
# #对比两天内同一时刻温度的变化情况
# x = [1, 2, 3, 4, 5, 6, 7]
# y1 = [2.4, 1.41, 1.269, 1.22, 1.21, 1.2, 1.2]     # AIDS
# y2 = [1.91, 1.13, 0.946, 0.93, 0.91, 0.89, 0.89]  # LINUX
# y3 = [2.67, 2.04, 1.724, 1.63, 1.52, 1.50, 1.50]  # IMDB
# y4 = [2.59, 2.13, 1.847, 1.73, 1.70, 1.71, 1.70]  # PTC
# #绘制折线图，添加数据点，设置点的大小
# plt.rc('font', family='Times New Roman')
#
# # * 表示绘制五角星；此处也可以不设置线条颜色，matplotlib会自动为线条添加不同的颜色
# plt.plot(x, y1, color="c", marker='*', markersize=10)
# plt.plot(x, y2, color='#F1404B', marker='x',markersize=8)
# plt.plot(x, y3, color='#F0DD92', marker='o',markersize=8)
# plt.plot(x, y4, color='#00A2DE', marker='^',markersize=8)
#
# # plt.title('温度对比折线图')  # 折线图标题
# plt.xlabel('Layer L', size=12)  # x轴标题
# plt.ylabel('MSE(e-3)',size=12)  # y轴标题
# #给图像添加注释，并设置样式
# # for a, b in zip(x, y1):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# # for a, b in zip(x, y2):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# # plt.yticks(np.arange(0, 3.25, step=0.25))
# # plt.ylim(ymin=-0.005)
#
# plt.xticks(np.arange(1, 8, step=1),size=12)
#
# plt.grid(ls='-.')  # 绘制背景线
# #绘制图例
# plt.legend(['AIDS', 'LINUX', 'IMDB', 'PTC'])
# plt.tight_layout()
# #显示图像
# plt.show()


######################## dimension ######################
# import matplotlib.pyplot as plt
# import numpy as np
# #对比两天内同一时刻温度的变化情况
# x = [8, 16, 24, 32, 64]
# y1 = [2.2, 1.269, 1.25, 1.22, 1.2]
# y2 = [1.8, 0.946, 0.89, 0.85, 0.8]
# y3 = [2.4, 1.724, 1.7, 1.65, 1.62]
# y4 = [2.7, 1.847, 1.5, 1.4, 1.3]
# #绘制折线图，添加数据点，设置点的大小
# plt.rc('font', family='Times New Roman')
# # * 表示绘制五角星；此处也可以不设置线条颜色，matplotlib会自动为线条添加不同的颜色
# plt.plot(x, y1, alpha=1, color="c", marker='*', markersize=10)
# plt.plot(x, y2, alpha=1, color='#F1404B', marker='x',markersize=8)
# plt.plot(x, y3, alpha=1, color='#F0DD92', marker='o',markersize=8)
# plt.plot(x, y4, alpha=1, color='#00A2DE', marker='^',markersize=8)
#
# # plt.title('温度对比折线图')  # 折线图标题
# plt.xlabel('Dimension d',size=12)  # x轴标题
# plt.ylabel('MSE(e-3)',size=12)  # y轴标题
# #给图像添加注释，并设置样式
# # for a, b in zip(x, y1):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# # for a, b in zip(x, y2):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.yticks(np.arange(0.5, 3.25, step=0.25))
# # plt.ylim(ymin=-0.005)
# plt.xticks(np.arange(8, 70, step=8),size=12)
# plt.grid(ls='-.')  # 绘制背景线
# #绘制图例
# plt.legend(['AIDS', 'LINUX', 'IMDB', 'PTC'])
# plt.tight_layout()
# #显示图像
# plt.show()

##################################GPSE Acc#####################
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
# # plt.figure(figsize=(15, 8))  # 设置画布
# x_index = np.arange(5)  # 确定label的位置 //箱数
# # 定义一个数字代表独立柱的宽度
# bar_width = 0.2
#
# x_data = ("GCNMean", "GCNMax", "SimGNN", "GraphSIM", "GPSE")
#
# """GED"""
# y1_data = (2.214,	3.423,	1.430,	1.880,	1.418)
# y2_data = (7.541,	6.341,	2.360,	1.076,	1.050)
# y3_data = (10,	    10,	    2.964,	1.924,	1.824)
# y4_data = (7.428,	8.329,	1.873,	1.889,	1.891)
#
# """MCS"""
# # y1_data = (6.234,	4.156,	3.433,	2.402,	2.400)
# # y2_data = (2.689,	2.170,	0.729,	3.164,	0.733)
# # y3_data = (10.457,	10.124,	2.451,	1.287,	1.226)
# # y4_data = (12.441,	13.845,	5.419,	3.975,	3.703)
#
#
# colors = ['indianred','#5A92AD', '#106D9C', '#00A2DE']
# # colors = ['indianred', '#106D9C', 'c', '#00A2DE']  8F88DA    8AC6D1  60A9A6
# plt.rc('font', family='Times New Roman')
#
# rects1 = plt.bar(x_index, y1_data, width=bar_width, alpha=0.8, color='#5A92AD', label="AIDS", edgecolor = "k", hatch= '///' )
# rects2 = plt.bar(x_index+bar_width, y2_data, width=bar_width, alpha=0.8, color="#106D9C", label="LINUX", edgecolor = "k",hatch= '\\\\\\')
# rects3 = plt.bar(x_index+(bar_width*2), y3_data, width=bar_width, alpha=0.8, color="#00A2DE", label="IMDB", edgecolor = "k",hatch= '++')
# rects4 = plt.bar(x_index+(bar_width*3), y4_data, width=bar_width, alpha=0.8, color="#8AC6D1", label="PTC", edgecolor = "k",hatch= 'xx')
# min_ylim, max_ylim = plt.ylim()
# min_xlim, max_xlim = plt.xlim()
# plt.xticks(x_index + bar_width + 0.1, x_data,size = 12)  # 设定x轴
# plt.legend()  # 显示图例
# # plt.title('Accuracy')
# plt.tight_layout()
# plt.ylabel('MSE',size = 12)
# # plt.grid(axis="y")
# plt.grid(ls='-.')  # 绘制背景线
# plt.tight_layout()
# path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))  # /src
# path = path + '/saved/figure.png'
# # plt.savefig(path, bbox_inches='tight')
# plt.show()


################################MBGSC  Acc###########################################
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
# # plt.figure(figsize=(15, 8))  # 设置画布
# x_index = np.arange(8)  # 确定label的位置
# # 定义一个数字代表独立柱的宽度
# bar_width = 0.2
# plt.rc('font', family='Times New Roman')
#
# x_data = ("GCNMean", "GCNMax", "SimGNN", "GraphSIM", "GMN", "MGMN","H2MN","MB-GSC")
#
# """GED"""
# y1_data = (2.214,	3.423,	1.430,	1.880,	4.310,	3.214,	2.211,  1.369)
# y2_data = (7.541,	6.341,	2.360,	1.076,	2.676,	5.259,	1.561,	1.046)
# y3_data = (10,	10,	2.964,	1.924,	3.210,	3.145,	2.232,	1.724)
# y4_data = (7.428,	8.329,	1.473,	1.889,	1.854,	3.650,	2.295,	1.647)
#
# """MCS"""
# # y1_data = (6.234,	4.156,	3.433,	2.402,	2.234,	3.719,	2.832,	2.213)
# # y2_data = (2.689,	2.170,	0.729,	3.164,	0.794,	0.739,	1.541,	0.753)
# # y3_data = (10.457,	10.124,	2.451,	1.287,	0.590,	5.128,	3.176,	0.546)
# # y4_data = (12.441,	13.845,	5.419,	3.975,	3.142,	2.765,	4.807,	2.653)
#
# colors = ['indianred','#5A92AD', '#106D9C', '#00A2DE']
# # colors = ['indianred', '#106D9C', 'c', '#00A2DE']  8F88DA    8AC6D1  60A9A6
#
# rects1 = plt.bar(x_index, y1_data, width=bar_width, alpha=0.8, color='#5A92AD', label="AIDS", edgecolor = "k", hatch='///')
# rects2 = plt.bar(x_index+bar_width, y2_data, width=bar_width, alpha=0.8, color="#106D9C", label="LINUX", edgecolor = "k",hatch='\\\\\\')
# rects3 = plt.bar(x_index+(bar_width*2), y3_data, width=bar_width, alpha=0.8, color="#00A2DE", label="IMDB", edgecolor = "k",hatch='++')
# rects4 = plt.bar(x_index+(bar_width*3), y4_data, width=bar_width, alpha=0.8, color="#8AC6D1", label="PTC", edgecolor = "k",hatch='xx')
# min_ylim, max_ylim = plt.ylim()
# min_xlim, max_xlim = plt.xlim()
# plt.xticks(x_index + bar_width + 0.1, x_data,size =11)  # 设定x轴
# plt.legend()  # 显示图例
# # plt.title('Accuracy')
# plt.ylabel('MSE',size=12)
# # plt.grid(axis="y")
# plt.grid(ls='-.')  # 绘制背景线
# plt.tight_layout()
#
# # # 保存图片
# # path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))  # /src
# # path = path + '/saved/figure.png'
# # plt.savefig(path, bbox_inches='tight')
# plt.show()


################################# Time ######################################################
# from matplotlib import pyplot as plt
#
# a = ["MB-GSC","H2MN", "MGMN", "GMN","GraphSIM", "SimGNN", "GCNMax", "GCNMean"]
#
# b_1 = [0.72, 0.5, 0.85, 0.82, 0.78, 0.43, 0.55, 0.55]   # AIDS
# b_2 = [0.74, 0.64, 0.71, 0.81, 0.77, 0.43, 0.51, 0.51]  # LINUX
# b_3 = [0.84, 0.55, 0.56, 0.83, 0.78, 0.46, 0.55, 0.55]  # IMDB
# b_4 = [0.85, 0.7, 0.86, 0.82, 0.80, 0.79, 0.73, 0.73]   # PTC
#
# height = 0.2
# a1 = list(range(len(a)))
# a2 = [i-height for i in a1]#坐标轴偏移
# a3 = [i-height*2 for i in a1]
# a4 = [i-height*3 for i in a1]
# plt.rc('font', family='Times New Roman')
#
# #绘图
# plt.barh(range(len(a)),b_1,height= height,label = "AIDS",color = "c", alpha=0.7)  # 4
# plt.barh(a2,b_2, height= height, label = "LINUX", color = "#F1404B", alpha=0.7) # 3
# plt.barh(a3,b_3,height= height,label = "IMDB",color = "#F0DD92", alpha=0.7) # 2
# plt.barh(a4,b_4,height= height,label = "PTC",color = "#00A2DE", alpha=0.7)  #1
#
# #绘制网格
# plt.grid(alpha = 0.4,ls='-.')
# # plt.grid()  # 绘制背景线
# #y轴坐标刻度标识
# plt.yticks(a2,a,size=12)
#
# #添加图例
# plt.legend()
#
# #添加横纵坐标，标题
# plt.xlabel("time(lg(t)×ms)",size=12)
# plt.xlim(xmax = 1.0)
# plt.tight_layout()
# #显示图形
# plt.show()

