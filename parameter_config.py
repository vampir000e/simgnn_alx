#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 20:19
# @Author  : LX Ann
# @FileName: parameter_config.py
# @Software: PyCharm
# @Email   : 578520080@qq.com

import argparse


# -----------运行时修改dataset, type, 以及load_model的路径---------------
# datasets = "AIDS700nef"
# datasets = "IMDBMulti"
# datasets = "PTC"
# datasets = "LINUX"
# datasets = "Test_PTC"

# type = "ged"
# type = "mcs"


def param_parser():
    parser = argparse.ArgumentParser(description="Run SimGNN_ALX.")

    parser.add_argument(
        "--dataset",
        nargs="?",
        default="AIDS700nef",
        help="Dataset name. Default is AIDS",
    )

    parser.add_argument(
        "--type",
        nargs="?",
        default="ged",
        help="ged or mcs. Default is ged",
    )

    parser.add_argument(
        "--epoch_num",
        type=int,
        default=100,
        help="Number of training epochs. Default is 100."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of graph pairs per batch. Default is 128.",
    )

    parser.add_argument(
        "--val",
        nargs="?",
        default=0.25,
        help="Validation rate. Default is 0.2",
    )

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

    parser.add_argument("--save-path",
                        type=str,
                        default=None,
                        help="Where to save the trained model")

    parser.add_argument("--load-path",
                        type=str,
                        default=None,
                        help="Load a pretrained model")

    parser.add_argument(
        "--notify",
        dest="notify",
        action="store_true",
        help="Send notification message when the code is finished (only Linux & Mac OS support).",
    )
    parser.set_defaults(notify=False)

    parser.add_argument("--perspectives", type=int, default=16, help='number of perspectives for node-graph matching')
    parser.add_argument("--match", type=str, default='node-graph', help="indicating the matching method")
    parser.add_argument("--global_flag", type=lambda x: (str(x).lower() == 'true'), default='True', help="Whether use global info ")
    parser.add_argument("--global_agg", type=str, default='fc_max_pool', help="aggregation function for global level gcn ")
    parser.add_argument("--hidden_size", type=int, default=16, help='hidden size for the graph-level embedding')
    parser.add_argument("--match_agg", type=str, default='bilstm', help="aggregator")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--filters", type=str, default='100_100_100', help="filters (neurons) for graph neural networks")
    parser.add_argument("--conv", type=str, default='gcn', help="one kind of graph neural networks")
    #parser.add_argument("--task", type=str, default='regression', help="classification/regression")

    return parser.parse_args()