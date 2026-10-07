import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from evaluation import compute_roc, compute_aupr, compute_mcc, acc_score, micro_score, compute_performance
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv,GATv2Conv,ResGatedGraphConv,TransformerConv,EGConv,FAConv,GINConv,GCN2Conv,GATConv,GPSConv,EGConv  #  ,SAGEConv

from torch import nn
from functools import partial

from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC


# from torch.nn import init
import torch_geometric

from graphsage import SAGEConv,WeightedSAGEConv,SAGEConv_1

device = "cuda" if torch.cuda.is_available() else "cpu"

from egnn_pytorch import EGNN_Network
import gvp
from gvp import GVP, GVPConvLayer, LayerNorm




class PPI_Model(nn.Module):
    def __init__(self,liner_dim=115,embbing=128):   #   114,121 115 92  115+64=179
        super(PPI_Model, self).__init__()
        self.multi_head_att=Attention_1(liner_dim,5)  #   多头注意力机制     头数必须可以整除输入特征维度
        self.multi_head_att2=Attention_1(liner_dim*2,10)  #   多头注意力机制     头数必须可以整除输入特征维度

        self.mlp1 = gcn_MLP(input_dim=liner_dim*1,  hidden_dim1=128,hidden_dim2=16,output_dim=1) 
        self.mlp2 = gcn_MLP(input_dim=liner_dim*2,  hidden_dim1=128,hidden_dim2=16,output_dim=1)


        ###  这是在修改节点的标量维度！
        ### node_dims= 115  #   6+                   edge_dims=32 +4     #   + 4
        # self.protein_model = GVPModel(node_in_dim=(115, 3), node_h_dim=(115, 3), edge_in_dim=(32 +4, 1),edge_h_dim=(32 +4, 1),num_layers=3, drop_rate=0.1)


        # egnn 最优的模块！！！
        self.egnnmodel=EGNN_Network(depth=4,dim=liner_dim,  edge_dim=0, norm_coors=True,coor_weights_clamp_value=2.).to(device)    #   4      num_nearest_neighbors=16,
        self.atom_model = atom_Model(int_dim=liner_dim,)
        self.CL_mi = contrast_learning(115,0.8,0.5)  # (args.cl_hidden, args.temperature, args.lambda_1)
        # self.softmax_layer = nn.Softmax(dim=1)

    # print(node_gvp_Scalar.detach().cpu().numpy().shape)
    def forward(self, node_number, node_features,bert_var,pseudo_dis, Res_matrix,        atom_onehot_embedding, edge_index_atom, res2atom_map,        coords_atom):
        bert_var = nn.BatchNorm1d(node_number)(bert_var.cpu()).to(device)# .to(device)   #   （90，33） 原始输入
        node_features = torch.cat((node_features,bert_var), dim=2)# 114+7+1024+768  1906 #   1,90,115


        egnn_output, egnn_all = self.egnnmodel(node_features, pseudo_dis)
        atom_embedding=self.atom_model(egnn_output,atom_onehot_embedding.squeeze(),edge_index_atom.squeeze(),res2atom_map.squeeze())

        # output = self.mlp1(egnn_output).squeeze(0)
        # return output,output

        ##    对比学习
        num_pos=int(node_number*0.1)    #   10      #   int(node_number*0.1)    #   10
        pos_miRNA_mask= construct_meta_pos(node_number,Res_matrix, num_pos)  # 获取正负样本对的掩码   #   后边这个数字是说 每一行选择多少个正样本


        ### 进行对比学习
        loss_cl, embedding = self.CL_mi(egnn_output.squeeze(0), atom_embedding, pos_miRNA_mask)
        ##去除对比学习
        # embedding=torch.cat((atom_embedding, egnn_output.squeeze(0)), dim=1)

        embedding = embedding.squeeze(0).unsqueeze(1)  # (90,1,726)
        embedding,_ =self.multi_head_att2(embedding)

        output = self.mlp2(embedding).squeeze(0)


        return output,loss_cl    # loss_cl




def atom2residue(atom_mat, residue_mat, res2atom_map):
    if len(atom_mat) != len(res2atom_map):
        return None
    else:
        new_atom_mat = torch.zeros((residue_mat.shape[0], atom_mat.shape[-1]))
        for a_id, a in enumerate(atom_mat):
            r_id = res2atom_map[a_id]
            new_atom_mat[r_id] += a
        return new_atom_mat

class atom_Model(nn.Module):
    def __init__(self, int_dim=115):
        super(atom_Model, self).__init__()
        self.int_dim = int_dim  #   int_dim

        self.gcn_x1 = SAGEConv(self.int_dim,self.int_dim)
        self.gcn_x2 = SAGEConv(self.int_dim,self.int_dim)
        self.gcn_x3 = SAGEConv(self.int_dim,self.int_dim)
        self.gcn_x4 = SAGEConv(self.int_dim,self.int_dim)

        # self.gcn_x1 = SAGEConv(37,37)
        # self.gcn_x2 = SAGEConv(37,37)
        # self.gcn_x3 = SAGEConv(37,37)
        # self.gcn_x4 = SAGEConv(37,37)

        self.ln=nn.Linear(37, self.int_dim)
        self.dp=nn.Dropout(0.2)

    def forward(self, resid_embedding, atom_onehot_embedding,edge_index_atom,res2atom_map):

        atom_onehot_embedding = self.ln(atom_onehot_embedding)

        atom_onehot_embedding = self.gcn_x1(atom_onehot_embedding,edge_index_atom)
        atom_onehot_embedding = self.gcn_x2(atom_onehot_embedding,edge_index_atom)
        atom_onehot_embedding = self.gcn_x3(atom_onehot_embedding,edge_index_atom)
        atom_onehot_embedding = self.gcn_x4(atom_onehot_embedding,edge_index_atom)

        feats_atom = atom2residue(atom_onehot_embedding.cpu(), resid_embedding.squeeze().cpu(), res2atom_map).cuda()     # 他就是原子特征

        # feats_atom = self.ln(feats_atom)


        return feats_atom  #   .squeeze()



class GVPModel(nn.Module):
    def __init__(self, node_in_dim, node_h_dim,edge_in_dim, edge_h_dim, num_layers=3, drop_rate=0.1,alpha=0.2):
        super().__init__()
        self.dp=nn.Dropout(p=0.3)
        self.alpha=alpha
        activations = (F.relu, None)
        activations_none = (None, None)

        self.node_norm_gvp_node = LayerNorm(node_in_dim)
        self.node_norm_gvp_edge = LayerNorm(edge_in_dim)

        self.GVP_gate_node = gvp.GVP(node_in_dim,node_in_dim, activations=activations_none, vector_gate=True) #   , vector_gate=True
        self.GVP_gate_edge = gvp.GVP(edge_in_dim, edge_in_dim, activations=activations_none) #    , vector_gate=True
        self.GVP_out = gvp.GVP(node_in_dim, node_in_dim, activations=activations_none, vector_gate=True)
        ns, nv = node_in_dim
        self.GVP_out3 = gvp.GVP((ns*3, nv), (ns*3, nv), activations=(None, None), vector_gate=True)

        self.node_norm_gvp = torch_geometric.nn.norm.LayerNorm(node_in_dim[0])
        self.GVPConv_gate = gvp.GVPConv(node_in_dim,node_in_dim, edge_in_dim,vector_gate=True ) #  activations=activations, vector_gate=True
        self.GVPConv_gate1 = gvp.GVPConv(node_in_dim,node_in_dim, edge_in_dim, vector_gate=True) # vector_gate=True
        self.GVPConv_gate2 = gvp.GVPConv(node_in_dim,node_in_dim, edge_in_dim,vector_gate=True ) # vector_gate=True
        self.GVPConv_gate3 = gvp.GVPConv(node_in_dim,node_in_dim, edge_in_dim,vector_gate=True ) # vector_gate=True
        self.GVPConvLayer_gate = gvp.GVPConvLayer(node_in_dim, edge_in_dim,vector_gate=True, drop_rate=drop_rate)   #   activations=activations,
        self.GVPConvLayer_gate1 = gvp.GVPConvLayer(node_in_dim, edge_in_dim, vector_gate=True, drop_rate=drop_rate)
        self.GVPConvLayer_gate2 = gvp.GVPConvLayer(node_in_dim, edge_in_dim,  vector_gate=True, drop_rate=drop_rate)




        self.GVP_gate_node1 = gvp.GVP(node_in_dim,node_in_dim, activations=activations_none, vector_gate=True) #   , vector_gate=True
        self.GVP_gate_edge1 = gvp.GVP(edge_in_dim, edge_in_dim, activations=activations_none) #    , vector_gate=True

        self.GVP_gate_node2 = gvp.GVP(node_in_dim,node_in_dim, activations=activations_none, vector_gate=True) #   , vector_gate=True
        self.GVP_gate_edge2 = gvp.GVP(edge_in_dim, edge_in_dim, activations=activations_none) #    , vector_gate=True

    def forward(self, node_s, node_v, edge_index1, edge_s, edge_v):
        ##  正则化
        node_s, node_v = self.node_norm_gvp_node((node_s, node_v))
        edge_s, edge_v = self.node_norm_gvp_edge((edge_s, edge_v))
        #   gvp预处理
        h_V = self.GVP_gate_node((node_s, node_v))  # h_V由两部分组成：标量和向量
        h_E = self.GVP_gate_edge((edge_s, edge_v))


        #   三层GVP卷积
        h_V0 = self.node_norm_gvp(h_V[0])
        h_V0, h_V1 = self.GVPConv_gate((h_V0, h_V[1]), edge_index1, h_E)    #   h_V[0]      h_V0
        # h_V0 = (1 - self.alpha) * h_V0 + self.alpha * h_V[0]
        # h_V0 = self.dp(h_V0)

        h_V0 = self.node_norm_gvp(h_V0)
        h_V0_1, h_V1_1 = self.GVPConv_gate1((h_V0, h_V1), edge_index1, h_E)
        # h_V0_1 = self.dp(h_V0_1)

        h_V0_1 = self.node_norm_gvp(h_V0_1)
        h_V0_2, h_V1_2 = self.GVPConv_gate2((h_V0_1, h_V1_1), edge_index1, h_E)
        # h_V0_2 = self.dp(h_V0_2)

        h_V0_2 = self.node_norm_gvp(h_V0_2)


        ### 将三层的输出进行拼接
        final_h_V0 = torch.cat((h_V0, h_V0_1, h_V0_2), dim=-1)  # 按最后一个维度拼接

        #   gvp后处理
        #h_V0_2, h_V1_2 = self.node_norm_gvp((h_V0_2, h_V1_2))
        node_gvp_Scalar, _ = self.GVP_out((h_V0_2, node_v))  # (90,121)
        # node_gvp_Scalar, _ = self.GVP_out3((final_h_V0, node_v))  # (90,121)

        # return node_gvp_Scalar, _
        return node_gvp_Scalar, final_h_V0




class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




class gcn_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1,hidden_dim2, output_dim):
        super(gcn_MLP, self).__init__()
        self.gc_layer = nn.Linear(input_dim, hidden_dim1)
        self.fc_layer1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, output_dim)   #   output_dim  2

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.gc_layer(x))
        x = self.dropout(x)
        x = F.relu(self.fc_layer1(x))
        x = self.dropout(x)
        x=self.output_layer(x)
        x = torch.sigmoid(x)
        # x = torch.softmax(self.output_layer(x),dim=1)
        return x




 ##########################################################################



import torch
import math


class Attention_1(nn.Module):  # Multi-headed Self-Attention Mechanism
    def __init__(self, hidden_size=128 * 2, num_attention_heads=8):
        super(Attention_1, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)

        self.sigmoid = torch.nn.Sigmoid()
        self.hidden_size = hidden_size

        self.layer_norm = torch.nn.LayerNorm(hidden_size).to(device)

        self.query = nn.Linear(hidden_size, self.hidden_size, bias=False).to(device)
        self.key = nn.Linear(hidden_size, self.hidden_size, bias=False).to(device)
        self.value = nn.Linear(hidden_size, self.hidden_size, bias=False).to(device)
        self.gate = nn.Linear(hidden_size, self.hidden_size).to(device)

    def transpose_for_scores(self, x):  # Divide the vector into two heads
        new_x_shape = x.size()[:-1] + (
        self.num_attention_heads, self.attention_head_size)  # [:-1]Left closed and right open not included-1
        x = x.view(*new_x_shape)
        # print(x.detach().cpu().numpy().shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, batch_hidden):
        # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x len

        # linear
        # key = torch.matmul(batch_hidden, self.weight) # b x len x hidden
        # batch_hidden=self.layer_norm(batch_hidden)
        query = self.query(batch_hidden)
        key = self.key(batch_hidden)
        value = self.key(batch_hidden)
        gate = self.sigmoid(self.gate(batch_hidden))
        #         key=batch_hidden
        #         query=batch_hidden
        #         print(key.shape)
        #         print(query.shape)
        # compute attention
        query = self.transpose_for_scores(query)  # batch,num_attention_heads,len,attention_head_size
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        outputs = torch.matmul(key, query.transpose(-1, -2))  # b x num_attention_heads*len*len

        attention_scores = outputs / math.sqrt(self.attention_head_size)  # (batch,num_attention_heads,len,len)

        attn_scores = F.softmax(attention_scores, dim=-1)  #

        # For an all-zero vector, -1e32 results in 1/len, -inf is nan, and the extra complement is 0
        #         masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        # sum weighted sources
        context_layer = torch.matmul(attn_scores, value)  # (batch,num_attention_heads,len,attention_head_size

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # (batch,n,num_attention_heads,attention_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size, 1)
        batch_outputs = context_layer.view(*new_context_layer_shape)  # (batch,n,all_head_size)
        # print(gate.shape)#32,33,128
        # print(batch_outputs.shape)#32,33,128,1
        batch_outputs = gate * batch_outputs.squeeze(3)

        batch_outputs = batch_outputs
        batch_outputs = torch.sum(batch_outputs, dim=1)
        # batch_outputs = batch_outputs[:,0]+batch_outputs[:,-1]

        return batch_outputs, attn_scores








import scipy.sparse as sp

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 构建元路径上的正样本对
def construct_meta_pos(node_number, adj,  num_pos):
    adj=adj.squeeze(0).detach().cpu().numpy()
    # print(adj.shape)

    target_pos = np.zeros((node_number, node_number))
    for i in range(len(adj)):
        target_pos[i, i] = 1        # 对角线是不是1

        one = adj[i].nonzero()[0]
        # print(one)
        # print(len(one))
        if len(one) > num_pos:
            oo = np.argsort(adj[i, one])       # 升序排列，取前边的
            sele = one[oo[:num_pos]]
            target_pos[i, sele] = 1
        else:
            target_pos[i, one] = 1

    #     # 设置对角线为 1
    # np.fill_diagonal(target_pos, 1)  # 填充对角线

    return torch.tensor(target_pos).to(device)

    # target_pos = sp.coo_matrix(target_pos)
    # target_pos = sparse_mx_to_torch_sparse_tensor(target_pos)
    #
    # # 返回的是一个掩码矩阵
    # # return torch.tensor(target_pos).to(device)
    # return target_pos



# 定义对比学习
class contrast_learning(nn.Module):
    def __init__(self, hidden, temperature, lambda_1):
        super(contrast_learning, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden)
        )
        self.temperature = temperature
        self.lambda_1 = lambda_1
        # 对权重矩阵进行初始化
        for fc in self.project:
            if isinstance(fc, nn.Linear):
                nn.init.xavier_normal_(fc.weight, gain=1.414)

    # 计算两个视图之间的相似性用于后续的损失函数
    def similarity(self, meta_view, sim_view):
        meta_view_norm = torch.norm(meta_view, dim=-1, keepdim=True)
        sim_view_norm = torch.norm(sim_view, dim=-1, keepdim=True)
        view_dot_fenzi = torch.mm(meta_view, sim_view.t())
        view_dot_fenmu = torch.mm(meta_view_norm, sim_view_norm.t())
        sim_matrix = torch.exp(view_dot_fenzi / view_dot_fenmu / self.temperature)
        return sim_matrix

    def forward(self, meta_, sim_, posSamplePairs):
        # 将特征经过一层线性层进行投影
        meta_project = self.project(meta_)
        sim_project = self.project(sim_)
        view_sim = self.similarity(meta_project, sim_project)
        view_sim_T = view_sim.t()

        view_sim = view_sim / (torch.sum(view_sim, dim=1).view(-1, 1) + 1e-8)
        loss_meta = -torch.log(view_sim.mul(posSamplePairs.to(device)).sum(dim=-1)).mean()

        view_sim_T = view_sim_T / (torch.sum(view_sim_T, dim=1).view(-1, 1) + 1e-8)
        loss_sim = -torch.log(view_sim_T.mul(posSamplePairs.to(device)).sum(dim=-1)).mean()

        return self.lambda_1 * loss_meta + (1 - self.lambda_1) * loss_sim, torch.cat((meta_project, sim_project), 1)
