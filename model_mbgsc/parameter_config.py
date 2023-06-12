#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 20:19
# @Author  : LX Ann
# @FileName: parameter_config.py
# @Software: PyCharm
# @Email   : 578520080@qq.com

import argparse

# -----------修改dataset---------------
# datasets = "AIDS700nef"
# datasets = "IMDBMulti"
# datasets = "PTC"
# datasets = "LINUX"
# datasets = "Test_PTC"
# type = "ged"
# type = "mcs"
def param_parser():
    parser = argparse.ArgumentParser(description="Run SimGNN_ALX.")
    parser.add_argument("--dataset", nargs="?", default="AIDS700nef", help="Dataset name. Default is Test_PTC",)
    parser.add_argument("--type", nargs="?", default="ged", help="ged or mcs. Default is ged",)
    parser.add_argument("--epoch_num", type=int, default=100,  help="Number of training epochs. Default is 40.")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of graph pairs per batch. Default is 128.",)
    parser.add_argument("--val", nargs="?", default=0.25, help="Validation rate. Default is 0.2",)
    parser.add_argument("--hist", dest="hist", action="store_true")
    parser.set_defaults(hist="hist")
    parser.add_argument("--ifDense_GCN", dest="ifDense_GCN", action="store_true")
    parser.set_defaults(ifDense_GCN=True)
    parser.add_argument("--random_id", dest="random_id", action="store_true")
    parser.set_defaults(random_id=True)
    parser.add_argument("--feedback", dest="feedback", action="store_true")
    parser.set_defaults(feedback=True)
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.set_defaults(plot=False)
    parser.add_argument("--synth", dest="synth", action="store_true")
    parser.set_defaults(synth=False)
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument("--save-path", type=str, default=None, help="Where to save the trained model")
    parser.add_argument("--load-path", type=str, default=None, help="Load a pretrained model")
    return parser.parse_args()