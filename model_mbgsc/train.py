#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 16:12
# @Author  : LX Ann
# @FileName: train.py
# @Software: PyCharm
# @Email   : 578520080@qq.com

import pickle
import random
import time
import numpy as np
import torch.optim
from tqdm import tqdm, trange
import torch.nn.functional as F
from mbgsc import SimGNN
from parameter_config import param_parser
from utils import tab_printer, random_id, write_log_file
from datetime import datetime
import torch.nn.functional as functional

class Trainer(object):

    def __init__(self, args, device):
        super(Trainer, self).__init__()
        self.args = args
        self.device = device
        self.prepare_for_train()

    def prepare_for_train(self):
        self.batch_size = self.args.batch_size
        self.epoch_num = self.args.epoch_num
        self.record = []  # 用来记录(train_loss, val_loss)
        self.random_id = self.args.random_id
        self.hist = self.args.hist
        self.ifDense_GCN = self.args.ifDense_GCN
        self.feedback = self.args.feedback
        self.dataset = self.args.dataset
        self.type = self.args.type
        self.val = self.args.val

        """载入数据并划分出 验证集"""
        # print("\nEnumerating unique labels.\n")

        path = "../../" + "datasets/" + self.dataset + "/" + self.type
        self.training_graphs = pickle.load(open(path + "/train_data.pickle", 'rb'))  #
        self.testing_graphs = pickle.load(open(path + "/test_data.pickle", 'rb'))

        # print(len(self.training_graphs)) # 37950

        random.shuffle(self.training_graphs)  # 训练集随机排序,然后随机划分
        L = len(self.training_graphs)
        # print(L)    # 37950    30360+7590 训练集 + 验证集
        div = int((1 - self.val) * L)
        self.val_graphs = self.training_graphs[div:L]  # 后20%验证集
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
        self.global_labels = {val: index for index, val in enumerate(self.global_labels)}
        self.labels = len(self.global_labels)  # 标签总数

        #######################################interface#####################################
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
        time = datetime.now()
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
                time_spent = datetime.now() - time
                time = datetime.now()
                # write_log_file(self.log_file, "{},{}, @ {}".format(batch, loss * 1000, time_spent))

            val_loss = self.validate()
            self.record.append((loss, val_loss))  # 记录(train_loss, val_loss)
            epochs.set_description("Epoch train_loss:[%g] val_loss:[%g]" % (round(loss, 5), round(val_loss, 5)))
            # write_log_file(self.log_file,
            #                "\n{},{}, spend time = {} @ {}".format(epoch, val_loss * 1000, time_spent,
            #                                                       datetime.now()))

            path = "../../datasets/" + self.dataset + "/" + self.type  # ged
            with open(path + '/train_error_graph.txt', "a") as train_error_writer:
                train_error_writer.write(str(epoch_counter) + ',' + str(round(loss, 6)) + '\n')
            train_error_writer.close()

            torch.save(self.model.state_dict(), path + '/model_state.pth')
            epoch_counter += 1
            self.score(epoch_counter)

            # self.train(epochs)

            # if (epoch+1) % 5 == 0:
            #     self.save_model("model"+str((epoch+1)/5)+".pkl")
            #     self.save_record("recordfile"+str((epoch+1)/5))

            # save_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))  # /src
            # save_path = save_path + "\\datasets\\" + self.datasets + "\\" + self.type
            # self.save_model(save_path + "\\model.pkl")
            # self.save_record(save_path + "\\recordfile")

    """每个epoch验证一次"""
    def validate(self):

        self.model.eval()
        losses = 0
        for data in self.val_graphs:
            data = self.transfer_to_torch(data)
            prediction = self.model(data)
            losses = losses + torch.nn.functional.mse_loss(data["target"], prediction[0])
        return losses.item()

    def score(self, epoch_counter):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        start_time = time.time()
        self.model.eval()

        scores = []
        losses = 0
        mae = 0
        rho = 0
        tau = 0
        pat10 = 0
        pat20 = 0
        # ground_truth = np.zeros((len(self.testing_graphs), len(self.training_graphs)))
        # prediction_mat = np.zeros((len(self.testing_graphs), len(self.training_graphs)))
        test_mse = 0
        test_rho = 0
        test_tau = 0
        # rho_list = []

        i = 0
        for data in tqdm(self.testing_graphs):  # data: test_graph
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            # prediction_mat[i] = prediction.detach().cpu().numpy()
            # ground_truth[i] = target.cpu().numpy()

            losses += torch.nn.functional.mse_loss(prediction[0], data["target"])
            scores.append(F.mse_loss(prediction, target, reduction="none"))
            mae = mae + torch.nn.functional.l1_loss(prediction[0], data["target"])
            # rho += calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i])
            # rho_list.append(
            #     calculate_ranking_correlation(
            #         spearmanr, prediction_mat[i], ground_truth[i]
            #     )
            # )
            # tau += calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i])
            # rho += metrics_spearmanr_rho(ground_truth[i], prediction_mat[i])
            # tau += metrics_kendall_tau(ground_truth[i], prediction_mat[i])
            # pat10 += calculate_prec_at_k(10, prediction_mat[i], ground_truth[i])
            # pat20 += calculate_prec_at_k(20, prediction_mat[i], ground_truth[i])
            i += 1

        # model_error = np.mean(scores).item()
        losses = losses / len(self.testing_graphs)
        mae = mae / len(self.testing_graphs)
        rho = rho / len(self.testing_graphs)
        # test_rho = np.mean(rho_list).item()
        tau = tau / len(self.testing_graphs)
        pat10 = pat10 / len(self.testing_graphs)
        pat20 = pat20 / len(self.testing_graphs)

        # res_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))  # /src
        # res_path = res_path + "\\datasets\\" + self.datasets + "\\" + self.type
        # with open(res_path + '/figures.txt', 'a') as f:
        #     print("\n--- %s seconds ---" % (time.time()), file=f)
        #     print("\nMSE: " + str(round(model_error, 5)) + ".", file=f)
        #     # print("\nModel test error: " + str(round(losses.item(), 5)) + ".", file=f)
        #     print("\nMAE: " + str(round(mae.item(), 5)) + ".", file=f)
        #     print("\nSpearman's rho: " + str(round(rho, 5)) + ".", file=f)
        #     print("\nKendall's tau: " + str(round(tau, 5)) + ".", file=f)
        #     print("p@10: " + str(round(pat10, 5)) + ".", file=f)
        #     print("p@20: " + str(round(pat20, 5)) + ".", file=f)
        #     print("\n")

        save_path = "../../datasets/" + self.dataset + "/" + self.type
        # print('\n\n >>>>>>>>>>>>>>>>>>\t' + str(model_error) + '\n')
        with open(save_path + "/test_error_graph.txt", "a") as test_error_writer:
            test_error_writer.write(str(epoch_counter) + ',' + str(losses) + '\n')
        test_error_writer.close()


        print("--- %s seconds ---" % (time.time() - start_time))
        # print("\nMSE: " + str(round(model_error, 5)) + ".")
        print("\nMSE: " + str(round(losses.item(), 5)) + ".")
        print("MAE: " + str(round(mae.item(), 5)) + ".")
        print("Spearman's rho: " + str(round(rho, 5)) + ".")
        # print("test_rho: " + str(round(test_rho, 5)) + ".")
        print("Kendall's tau: " + str(round(tau, 5)) + ".")
        print("p@10: " + str(round(pat10, 5)) + ".")
        print("p@20: " + str(round(pat20, 5)) + ".")

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

    """将读入的文件转化成网络能接受的形式"""
    def transfer_to_torch(self, data):
        new_data = dict()
        tmp_edges_1 = data["graph_1"]
        tmp_edges_2 = data["graph_2"]
        tmp_labels_1 = data["labels_1"]
        tmp_labels_2 = data["labels_2"]

        if self.random_id is True:
            tmp_edges_1, tmp_labels_1 = random_id(tmp_edges_1, tmp_labels_1)
            tmp_edges_2, tmp_labels_2 = random_id(tmp_edges_2, tmp_labels_2)

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
        norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
        new_data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float().to(self.device)

        return new_data

    def training_batch_predication(self, batch_feature_1, batch_adjacent_1, batch_mask_1, batch_feature_2,
                                   batch_adjacent_2, batch_mask_2, ged_pairs):

        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.batch_pairs_predication(batch_feature_1, batch_adjacent_1, batch_mask_1, batch_feature_2,
                                                   batch_adjacent_2, batch_mask_2)
        trues = torch.from_numpy(np.array(ged_pairs, dtype=np.float32)).to(self.device)

        loss = functional.mse_loss(predictions, trues)
        loss.backward()
        self.optimizer.step()

        return loss.item(), torch.stack((trues, predictions), 1)

if __name__ == '__main__':
    import os
    args = param_parser()
    d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index

    # tab_printer(args)
    trainer = Trainer(args, device=d)
    trainer.fit()
    trainer.score()

