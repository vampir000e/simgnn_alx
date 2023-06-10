#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/10 17:02
# @Author  : LX Ann
# @FileName: utils.py
# @Software: PyCharm
# @Email   : 578520080@qq.com

import os
from scipy import stats

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 'Make dirs of # {} '.format(directory)
    else:
        return "the dirs already exist! Cannot be created"

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
