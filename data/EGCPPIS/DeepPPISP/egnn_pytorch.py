import torch
from torch import nn, einsum, broadcast_tensors

from einops import rearrange, repeat


def exists(val):
    return val is not None


# swish activation fallback

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_


# helper classes

# this follows the same strategy for normalization as done in SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95

class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale



class EGNN(nn.Module):
    def __init__(self, dim, edge_dim=0, m_dim=16, fourier_features=0, num_nearest_neighbors=0, dropout=0.1,     #   dropout=0  0.egnn     # num_nearest_neighbors=0 16
                 init_eps=1e-3, norm_feats=False, norm_coors=False, norm_coors_scale_init=1e-2, update_feats=True,
                 update_coors=True, only_sparse_neighbors=False, valid_radius=float('inf'), m_pool_method='sum',
                 soft_edges=False, coor_weights_clamp_value=None):
        ## ##  dim=115, edge_dim=(4 + 0), norm_feats=True, norm_coors=True,coor_weights_clamp_value=2.

        super().__init__()
        assert m_pool_method in {'sum', 'mean'}, 'pool method must be either sum or mean'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'


        self.fourier_features = fourier_features    # 0

        edge_input_dim = (fourier_features * 2) + (dim * 2) + edge_dim + 1
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(nn.Linear(edge_input_dim, edge_input_dim * 2), dropout, SiLU(),
                                      nn.Linear(edge_input_dim * 2, m_dim), SiLU())

        self.edge_gate = nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid()) if soft_edges else None


        self.node_norm = nn.LayerNorm(dim) if norm_feats else nn.Identity()
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()

        self.m_pool_method = m_pool_method

        self.node_mlp = nn.Sequential(nn.Linear(dim + m_dim, dim * 2), dropout, SiLU(),
                                      nn.Linear(dim * 2, dim), ) if update_feats else None

        self.coors_mlp = nn.Sequential(nn.Linear(m_dim, m_dim * 4), dropout, SiLU(),
                                       nn.Linear(m_dim * 4, 1)) if update_coors else None


        self.num_nearest_neighbors = num_nearest_neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_radius = valid_radius

        self.coor_weights_clamp_value = coor_weights_clamp_value


        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.normal_(module.weight, std=self.init_eps)

    def forward(self, feats, coors, edges=None, mask=None, adj_mat=None):

        # 计算相对位置矩阵
        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)
        #   计算节点间相对坐标与距离，下边两个是 计算节点之间的相坐标 欧式距离。


        feats_j = rearrange(feats, 'b j d -> b () j d')

        feats_i = rearrange(feats, 'b i d -> b i () d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)

        if exists(edges):
            edge_input = torch.cat((edge_input, edges), dim=-1)
        #   (egnn, 90, 90, 231)        (egnn, 90,90, 4)

        m_ij = self.edge_mlp(edge_input)

        if exists(self.edge_gate):
            m_ij = m_ij * self.edge_gate(m_ij)



        if exists(self.coors_mlp):
            coor_weights = self.coors_mlp(m_ij)
            coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

            rel_coors = self.coors_norm(rel_coors)


            if exists(self.coor_weights_clamp_value):   #   有用
                clamp_value = self.coor_weights_clamp_value
                coor_weights.clamp_(min=-clamp_value, max=clamp_value)

            coors_out = einsum('b i j, b i j c -> b i c', coor_weights, rel_coors) + coors
        else:
            coors_out = coors



        if exists(self.node_mlp):
            if self.m_pool_method == 'mean':
                m_i = m_ij.mean(dim=-2)
            elif self.m_pool_method == 'sum':
                m_i = m_ij.sum(dim=-2)

            normed_feats = self.node_norm(feats)
            node_mlp_input = torch.cat((normed_feats, m_i), dim=-1)
            node_out = self.node_mlp(node_mlp_input) + feats
        else:
            node_out = feats

        return node_out, coors_out

'''
adj_dim 决定了邻接矩阵的嵌入维度，当图的节点关系不仅限于一阶邻接时，可以将不同阶数的邻接关系编码进邻接矩阵。

'''

import torch_geometric
class EGNN_Network(nn.Module):
    def __init__(self, *, depth, dim, edge_dim=0,  num_nearest_neighbors=16,num_edge_tokens=None, num_positions=None,
                 num_adj_degrees=None, adj_dim=0, global_linear_attn_every=0, global_linear_attn_heads=8,
                 global_linear_attn_dim_head=64, num_global_tokens=4, **kwargs):    # kwargs里边包含了  norm_coors=True,coor_weights_clamp_value=2.
        super().__init__()
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than egnn'

        #   如下属于暂未使用的部分，
        self.num_positions = num_positions
        self.pos_emb = nn.Embedding(num_positions, dim) if exists(num_positions) else None
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None

        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None
        adj_dim = adj_dim if exists(num_adj_degrees) else 0


        has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        if has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))

        self.num_nearest_neighbors = num_nearest_neighbors


        ###     这里是有用的
        self.has_edges = edge_dim > 0
        edge_dim = edge_dim if self.has_edges else 0

        self.layers = nn.ModuleList([])
        for ind in range(depth):    #   (0)None  (egnn)三层EGNN
            self.layers.append(nn.ModuleList([EGNN(dim=dim, edge_dim=(edge_dim + adj_dim), norm_feats=True, **kwargs)]))  # edge_dim=4+0

        # print(self.layers)
        self.Antilayer=torch_geometric.nn.conv.AntiSymmetricConv(in_channels=115, act='tanh')

        self.instanceNorm = torch_geometric.nn.norm.InstanceNorm(115)

        self.dp=nn.Dropout(0.2)

    def forward(self, feats, coors, edges=None, adj_mat=None,mask=None, return_coor_changes=False):
        # b, device = feats.shape[0], feats.device

        residual_feats=feats
        feats_list = []
        h0=feats
        lamda = 1.1
        alpha = 0.1
        ii=1

        for egnn in self.layers:
            feats, coors = egnn[0](feats, coors,edges=edges, adj_mat=adj_mat, mask=mask)

            residual_feats = feats  #   残差连接
            feats_list.append(feats)

        ### 最后将所有层的 feats cat 起来
        feats_all = torch.cat(feats_list, dim=2)  # 选择合适的维度进行连接

        # 输出维度为(B,N,dim), (B,N,d)
        return feats, feats_all




# 将 feats_cat 追加写入 txt 文件
def append_feats_to_txt(feats_cat, file_path="feats_output.txt"):
    # 将张量转换为 numpy 数组（便于保存为字符串）
    feats_cat_np = feats_cat.detach().cpu().numpy()

    # 打开文件并追加写入
    with open(file_path, "a") as f:
        for row in feats_cat_np:
            # 将每一行转换为字符串并写入文件
            line = " ".join(map(str, row))  # 将每一行元素转换为字符串并以空格分隔
            f.write(line + "\n")
            f.write(line + "\n")
            f.write(line + "\n")



















