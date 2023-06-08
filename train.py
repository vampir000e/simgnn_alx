#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/8 16:12
# @Author  : LX Ann
# @FileName: train.py
# @Software: PyCharm
# @Email   : 578520080@qq.com

import pickle
import random

import numpy as np
import torch.optim
from tqdm import tqdm, trange
from simgnn import SimGNN
from parameter_config import param_parser
from extra import tab_printer, random_id


class Trainer(object):

    def __init__(self, args, device):
        super(Trainer, self).__init__()
        self.args = args
        self.device = device
        self.prepare_for_train()

    def prepare_for_train(self):
        self.batch_size = self.args.batch_size
        self.epoch_num = self.args.epoch_num
        self.record = [] # 用来记录(train_loss, val_loss)
        self.random_id = self.args.random_id
        self.hist = self.args.hist
        self.ifDense_GCN = self.args.ifDense_GCN
        self.feedback = self.args.feedback
        self.dataset = self.args.dataset
        self.type = self.args.type
        self.val = self.args.val

        """载入数据并划分出验证集"""
        print("\nEnumerating unique labels.\n")

        path = "../datasets/" + self.dataset + "/" + self.type
        self.training_graphs = pickle.load(open(path + "/train_data.pickle", 'rb'))
        self.testing_graphs = pickle.load(open(path + "/test_data.pickle", 'rb'))

        random.shuffle(self.training_graphs) #训练集随机排序，然后随机划分
        L = len(self.training_graphs)
        # print(L)    # 37950    30360+7590 训练集 + 验证集
        div = int((1 - self.val) * L)
        self.val_graphs = self.training_graphs[div:L] #后20%验证集
        self.training_graphs = self.training_graphs[0:div]

        graph_pairs = self.training_graphs + self.testing_graphs  # 30360 + 18975 = 49335
        # print("-------")
        # print (len(graph_pairs))   # 49335   train_graph + test_graph  30360 + 18975 = 49335
        # print(len(self.training_graphs))  # 30360  37950 * 0.8 = 30360
        # print(len(self.val_graphs))       # 7590   37950 * 0.2 = 18975
        # print(len(self.testing_graphs))  # 18975

        """求出一共的特征数量"""
        self.global_labels = set()
        for data in tqdm(graph_pairs):
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = list(self.global_labels)
        self.global_labels = {val:index for index, val in enumerate(self.global_labels)}
        self.labels = len(self.global_labels) #标签总数

        """interface"""
        self.model = SimGNN(self.args, self.labels, self.device).to(self.device)
        # self.model = SimGNN_graphsim(self.labels, self.hist, self.ifDense_GCN, self.feedback)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")

        epoch_counter = 0
        loss = 0
        bool = False
        self.model.train()
        epochs = trange(self.epoch_num, leave=True, desc="Epoch")

        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            self.epoch_loss = 0
            self.node_processed = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                self.epoch_loss = self.epoch_loss + self.process_batch(batch)
                self.node_processed = self.node_processed + len(batch)
                loss = self.epoch_loss / self.node_processed
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 6))

            val_loss = self.validate()
            self.record.append((loss, val_loss)) #记录(train_loss, validate_loss)
            epochs.set_description("Epoch train_loss:[%g] val_loss:[%g]" % (round(loss, 5), round(val_loss, 5)))

            path = "../datasets/" + self.dataset + "/" + self.type
            with open(path + 'train_error_graph.txt', 'a') as train_error_writer:
                train_error_writer.write(str(epoch_counter) + ',' + str(round(loss, 6)) + '\n')
            train_error_writer.close()

            torch.save(self.model.state_dict(), path + '/model_state.pth')
            epoch_counter += 1
            self.score(epoch_counter)

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph+self.args.batch_size])
        return batches

    """每个batch调用一次"""
    def process_batch(self, batch):
        self.optimizer.zero_grad() #梯度清0
        losses = 0
        for data in batch:
            data = self.transfer_to_torch(data)
            prediction = self.model(data).to(self.device)
            tmp = torch.nn.functional.mse_loss(data["target"], prediction[0])
            losses = losses + tmp
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def transfer_to_torch(self, data):
        new_data = dict()
        tmp_edges_1 = data["graph_1"]
        tmp_edged_2 = data["graph_2"]
        tmp_labels_1 = data["labels_1"]
        tmp_labels_2 = data["labels_2"]

        if self.random_id is True:
            tmp_edges_1, tmp_labels_1 = random_id(tmp_edges_1, tmp_labels_1)
            tmp_edges_2, tmp_labels_2 = random_id(tmp_edged_2, tmp_labels_2)

        edges_1 = tmp_edges_1 + [[y, x] for x, y in tmp_edges_1]
        edges_2 = tmp_edges_2 + [[y, x] for x, y in tmp_edges_2]

        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        features_1, features_2 = [], []
        for n in tmp_labels_1:
            features_1.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        for n in tmp_labels_2:
            features_2.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))

        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2
        new_data["features_1"] = features_1
        new_data["features_2"] = features_2
        norm_ged = data["ged"] / (0.5 * (len(data["labels_1"]) + len(data["labels_2"])))
        new_data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float().to(self.device)

        return new_data


if __name__ == '__main__':
    import os
    args = param_parser()
    d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index

    tab_printer(args)
    trainer = Trainer(args, device=d)
    trainer.fit()

    if args.notify:
        import os
        import sys
        if sys.platform == "linux":
            os.system('notify-send SimGNN "Program is finished."')
        elif sys.platform == "posix":
            os.system(
                """
                    osascript -e 'display notification :"SimGNN" with title "Program is finished."'
                """
            )
        else:
            raise NotImplementedError("No notification support for this OS.")