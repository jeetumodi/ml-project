import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from evaluation import compute_roc, compute_aupr, compute_mcc, acc_score, micro_score, compute_performance
import os
import time
from torch.nn.parameter import Parameter
from utils_Test import ProDataset
from torch_geometric.utils import dense_to_sparse
import pandas as pd
import random


from Model_alpha import AverageMeter, PPI_Model

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = "cuda" if torch.cuda.is_available() else "cpu"


early_stopping = False
use_pLM = False

map_cutoff = 16  # 14
THREADHOLD = 0.4
Freq = 100000000


def eval_epoch(model, loader, print_freq=Freq, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    global THREADHOLD
    # Model on eval mode
    model.eval()

    all_trues = []
    all_preds = []
    end = time.time()

    for batch_idx, (
            seq_data, dssp_data, atom_onehot_embedding, edge_index_atom, res2atom, bert_data, pseudo_dis, dis_data,
            Res_matrix, node_features, label, coords_atom, seq1) in enumerate(loader):
        node_number = np.array(seq_data)[0].shape[0]

        # 将密集矩阵转换为稀疏表示
        edge_index1, _ = dense_to_sparse(dis_data)
        edge_index = edge_index1.to(torch.int64)  # .t()

        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():

                node_features = torch.autograd.Variable(node_features.cuda().float())
                bert_var = torch.autograd.Variable(bert_data.cuda().float())
                pseudo_dis = torch.autograd.Variable(pseudo_dis.cuda().float())  # .float())
                Res_matrix = torch.autograd.Variable(Res_matrix.cuda().float())  # .float()

                atom_onehot_embedding = torch.autograd.Variable(atom_onehot_embedding.cuda().float())  # .float()
                edge_index_atom = torch.autograd.Variable(edge_index_atom.cuda())
                coords_atom = torch.autograd.Variable(coords_atom.cuda().float())
                res2atom = torch.autograd.Variable(res2atom.cuda())

                target_var = torch.autograd.Variable(label.cuda())  # .float()
                seq_var = torch.autograd.Variable(seq_data.cuda().float())
                print('111')

            else:
                node_features = torch.autograd.Variable(node_features.float())
                bert_var = torch.autograd.Variable(bert_data.float())
                pseudo_dis = torch.autograd.Variable(pseudo_dis.float())
                Res_matrix = torch.autograd.Variable(Res_matrix).float()  # .float()

                atom_onehot_embedding = torch.autograd.Variable(atom_onehot_embedding).float()  # .float()
                edge_index_atom = torch.autograd.Variable(edge_index_atom)
                coords_atom = torch.autograd.Variable(coords_atom)
                res2atom = torch.autograd.Variable(res2atom.float())

                target_var = torch.autograd.Variable(label)  # .float()
                seq_var = torch.autograd.Variable(seq_data.float())
                print('222')

        # compute output
        output, cl_loss = model(node_number, node_features, bert_var, pseudo_dis, Res_matrix, atom_onehot_embedding,
                                edge_index_atom, res2atom, coords_atom, )

        output = output.transpose(0, 1)

        loss = torch.nn.functional.binary_cross_entropy(output, target_var.float())  # .cuda()
        loss = loss + cl_loss *0.1


        # measure accuracy and record loss
        batch_size = label.size(0)
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        all_trues.append(label.numpy())
        all_preds.append(output.data.cpu().numpy())  # 使用教师模型的输出：output

    trues, preds, preds_pLM = [], [], []
    for i in range(len(all_trues)):
        trues.append(np.array(all_trues[i]).flatten())
        preds.append(np.array(all_preds[i]).flatten())

    all_trues = np.concatenate(trues, axis=0)  # np.array(all_trues)
    all_preds = np.concatenate(preds, axis=0)


    auc ,tprs= compute_roc(all_preds, all_trues)
    aupr, mean_precision, mean_recall = compute_aupr(all_preds, all_trues)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_preds, all_trues)

    if predictions_max is None:
        predictions_max = np.zeros_like(all_trues, dtype=float)
        print('HH' * 100)

    acc_val = acc_score(predictions_max, all_trues)
    mcc = compute_mcc(predictions_max, all_trues)


    #   保存绘制ROC曲线和PR曲线的值
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_fpr = np.linspace(0, 1, 100)
    dataframe = pd.DataFrame({'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr})
    dataframe.to_csv("model_save/ROC_data.csv", index=False, sep=',')


    dataframe = pd.DataFrame({'mean_precision': mean_precision, 'mean_recall': mean_recall})
    dataframe.to_csv("model_save/PR_data.csv", index=False, sep=',')

    
    return batch_time.avg, losses.avg, acc_val, f_max, p_max, r_max, auc, aupr, t_max, mcc


def train(model, train_data_set, valid_data_set, n_epochs=10, batch_size=32, seed=None, save_path="results/"):
    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=1,
                                               pin_memory=(torch.cuda.is_available()),
                                               num_workers=0)  # , drop_last=True
    valid_loader = torch.utils.data.DataLoader(valid_data_set, batch_size=1,
                                               pin_memory=(torch.cuda.is_available()),
                                               num_workers=0)  # , drop_last=True

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()


    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model

    _, valid_loss, acc, f_max, p_max, r_max, auc, aupr, t_max, mcc = eval_epoch(
            model=model_wrapper,
            loader=valid_loader,
            is_test=(not valid_loader)
        )

    
    # f.write('epoch,valid_loss,acc,F_value, precision,recall,auc,aupr,mcc,threadhold\n')
    with open('results_test315.txt', 'a') as file:  # 使用 'a' 模式来追加内容，而不是覆盖
        file.write(
            f'{valid_loss}, {acc}, {f_max}, {p_max}, {r_max}, {auc}, {aupr}, {mcc}, {t_max}\n')
    print(
            'valid_loss:%0.5f:\nacc:%0.6f,F_value:%0.6f, precision:%0.6f,recall:%0.6f,auc:%0.6f,aupr:%0.6f,mcc:%0.6f,threadhold:%0.6f' % (
                valid_loss, acc, f_max, p_max, r_max, auc, aupr, mcc, t_max))

    print()
    print()


def main():
    file_path_train = "Dataset/Test_60.fa"
    file_path_varify = "Dataset/GraphPPIS/Test_315-28.fa"
    # file_path_varify = "Dataset/GraphPPIS/UBtest_31-6.fa"
    train_dataSet = ProDataset(file_path_train)
    varify_dataSet = ProDataset(file_path_varify)

    model = PPI_Model()
    model.load_state_dict(torch.load("model_save/best_model.pth"))

    # Train the model
    train(model=model, train_data_set=train_dataSet, valid_data_set=varify_dataSet, n_epochs=60, batch_size=32)
    print('Done!!!')



if __name__ == '__main__':
    main()




