#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 16:34
# @Author  : LX Ann
# @FileName: parameter_config.py
# @Software: PyCharm
# @Email   : 578520080@qq.com

import argparse
parser = argparse.ArgumentParser(description="Graph-Pair Similarity Embedding Network for the Graph-Graph Regression task")
parser.add_argument("--data-dir", type=str, default="../../datasets", help='root directory for the data')
parser.add_argument('--dataset', type=str, default="AIDS700nef", help='indicate the specific dataset')  # datasets = "AIDS700nef" "IMDBMulti" "PTC" "LINUX"
parser.add_argument("--conv", type=str, default='gcn', help="one kind of graph neural networks")
parser.add_argument('--iterations', type=int, default= 10, help='number of training epochs')   # default=1000 900 100
parser.add_argument('--iter_val_start', type=int, default=9)  #
parser.add_argument('--iter_val_every', type=int, default=1)  #
parser.add_argument("--batch_size", type=int, default=128, help="Number of graph pairs per batch.")
parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
parser.add_argument('--log_path', type=str, default='./GEDLogs/', help='path for log file')
parser.add_argument("--filters", type=str, default='100_100_100', help="filters (neurons) for graph neural networks")
parser.add_argument("--perspectives", type=int, default=100, help='number of perspectives for matching')
parser.add_argument("--match_agg", type=str, default='bilstm', help="aggregator")
parser.add_argument("--hidden_size", type=int, default=100, help='hidden size for the graph-level embedding')
parser.add_argument("--global_flag", type=lambda x: (str(x).lower() == 'true'), default='True', help="Whether use global info ")
parser.add_argument("--global_agg", type=str, default='fc_max_pool', help="aggregation function for global level gcn ")
parser.add_argument('--inclusive', type=lambda x: (str(x).lower() == 'true'), default='True', help='True')
parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
parser.add_argument('--repeat_run', type=int, default=1, help='indicated the index of repeat run')


ged_args = parser.parse_args()