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
from utils import ProDataset
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


def train_epoch(model, loader, optimizer, epoch, all_epochs, print_freq=Freq):
    batch_time = AverageMeter()
    losses = AverageMeter()

    global THREADHOLD
    # Model on train mode
    model.train()

    all_preds = []
    all_trues = []
    end = time.time()

    for batch_idx, (seq_data, dssp_data,    atom_onehot_embedding, edge_index_atom, res2atom,
        bert_data, pseudo_dis, dis_data, Res_matrix,node_features, label, coords_atom, seq1) in enumerate(loader):


        # dssp_data=dssp_data[:,:,:9] #   仅仅使用9维的二级结构
        # bert_data=bert_data[:,:,512:]
        node_number = np.array(seq_data)[0].shape[0]

        # 将密集矩阵转换为稀疏表示
        edge_index1, _ = dense_to_sparse(dis_data)
        edge_index = edge_index1.to(torch.int64)  # .t()

        # g = dgl.graph((edge_index1[0],edge_index1[1])).to('cuda:0')

        # edge_attr=cal_edge_attr(edge_index1,pseudo_dis.squeeze(0),node_features)
        # edge_attr=torch.tensor(edge_attr)
        # edge_attr = edge_attr.permute(1, 0) #   [4565,2]

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


        # compute output
        output, cl_loss = model(node_number, node_features, bert_var, pseudo_dis,  Res_matrix,     atom_onehot_embedding, edge_index_atom, res2atom,  coords_atom,)


        all_trues.append(target_var.float().detach().cpu().numpy())
        all_preds.append(output.detach().cpu().numpy())

        output = output.transpose(0, 1)
        loss = torch.nn.functional.binary_cross_entropy(output, target_var.float())  # .cuda()
        loss = loss + cl_loss *0.1


        # measure accuracy and record loss
        batch_size = label.size(0)
        pred_out = output.ge(THREADHOLD)

        # print(pred_out)
        pred_out = pred_out.data.cpu().numpy()
        target_var = target_var.data.cpu().numpy()

        # print(pred_out.shape,target_var.shape)
        MiP, MiR, MiF, PNum, RNum = micro_score(pred_out,
                                                target_var)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, all_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'f_max:%.6f' % (MiP),
                'p_max:%.6f' % (MiR),
                'r_max:%.6f' % (MiF),
                't_max:%.2f' % (PNum)
            ])
            print('\n', res)

    return _, losses.avg,


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

        # dssp_data=dssp_data[:,:,:9] #   仅仅使用9维的二级结构
        # bert_data=bert_data[:,:,512:]
        node_number = np.array(seq_data)[0].shape[0]

        # 将密集矩阵转换为稀疏表示
        edge_index1, _ = dense_to_sparse(dis_data)
        edge_index = edge_index1.to(torch.int64)  # .t()

        # g = dgl.graph((edge_index1[0],edge_index1[1])).to('cuda:0')

        # edge_attr=cal_edge_attr(edge_index1,pseudo_dis.squeeze(0),node_features)
        # edge_attr=torch.tensor(edge_attr)
        # edge_attr = edge_attr.permute(1, 0) #   [4565,2]

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


    # Train model
    best_aupr = 0
    threadhold = 0
    count = 0

    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=0.0008, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, min_lr=1e-6)  # 5e-6   1e-6

    for epoch in range(n_epochs):
        # # # # Optimizer
        # if epoch < 20:
        #     optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=0.0008)  # lr=0.0008
        # elif epoch < 30:
        #     optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=0.0004)  # lr=0.0004
        # elif epoch < 40:
        #     optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=0.0002)  # lr=0001
        # elif epoch < 50:
        #     optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=0.0001)  # lr=00005
        # else:
        #     optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=0.00005)  # lr=0.00001

        # optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=0.0004)


        # _, train_loss = train_epoch(
        _, train_loss = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            all_epochs=n_epochs,
        )

        _, valid_loss, acc, f_max, p_max, r_max, auc, aupr, t_max, mcc = eval_epoch(
            model=model_wrapper,
            loader=valid_loader,
            is_test=(not valid_loader)
        )

        ####### 更新学习率
        scheduler.step(aupr)  # aupr_lr
        print(f'Epoch {epoch}, Current Learning Rate:', optimizer.param_groups[0]['lr'])


        # f.write('epoch,valid_loss,acc,F_value, precision,recall,auc,aupr,mcc,threadhold\n')
        with open('results.txt', 'a') as file:  # 使用 'a' 模式来追加内容，而不是覆盖
            file.write(
                f'{epoch + 1}, {valid_loss}, {acc}, {f_max}, {p_max}, {r_max}, {auc}, {aupr}, {mcc}, {t_max}\n')

        print(
            'epoch:%03d,valid_loss:%0.5f:\nacc:%0.6f,F_value:%0.6f, precision:%0.6f,recall:%0.6f,auc:%0.6f,aupr:%0.6f,mcc:%0.6f,threadhold:%0.6f' % (
                (epoch + 1), valid_loss, acc, f_max, p_max, r_max, auc, aupr, mcc, t_max))

        if aupr > best_aupr:  # chatgpt说 f_max是F1分数，但我也不知道为什么要根据这个选择最优模型。所以我换成了aupr
            count = 0
            best_aupr = aupr
            THREADHOLD = t_max
            # print("new best F_value:{0}(threadhold:{1})".format(f_max, THREADHOLD))
            print('-' * 20, "new best AUPR:{0}(threadhold:{1})".format(aupr, THREADHOLD), '-' * 20)
            torch.save(model.state_dict(), "model_save/best_model_"+str(epoch)+"_"+str(best_aupr)+".pth")
            # model.load_state_dict(torch.load("best_model.pth"))
        else:
            count += 1
            if count >= 250:
                return None

        print()
        print()


def main():
    # set_seed(1)
    # test_randomness()

    file_path_train = "Dataset/Train_335.fa"
    file_path_varify = "Dataset/Test_60.fa"
    train_dataSet = ProDataset(file_path_train)
    varify_dataSet = ProDataset(file_path_varify)

    model = PPI_Model()

    # Train the model
    train(model=model, train_data_set=train_dataSet, valid_data_set=varify_dataSet, n_epochs=60, batch_size=32)
    print('Done!!!')




def set_seed(seed=100000):
    # 设置Python的随机种子
    random.seed(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    # 如果使用GPU，也需要固定GPU的随机数种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    # 确保GPU上的随机性固定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 测试随机性是否一致
def test_randomness():
    print("Random:", random.random())
    print("Numpy Random:", np.random.rand())
    print("Torch Random:", torch.rand(1))

    if torch.cuda.is_available():
        print("Torch CUDA Random:", torch.cuda.FloatTensor(1).uniform_())
    else:
        print("No CUDA")


if __name__ == '__main__':
    main()




