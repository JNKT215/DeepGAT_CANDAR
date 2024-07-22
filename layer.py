import math
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    softmax
)
from torch_geometric.nn.inits import zeros


class DeepGATConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_class: int, heads: int = 1,
                 concat: bool = True,dropout: float = 0.0, 
                 add_self_loops: bool = True,
                 bias: bool = True,attention_type: str = 'SD',class_num: str = 'Single',oracle_attention: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.alpha_ = None
        self.h = None
        self.attention_type=attention_type
        self.class_num= class_num
        self.oracle_attention = oracle_attention
        self.oracle_alpha_ = None
        self.num_class = num_class

        self.lin = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
        self.lin_c = Linear(in_channels, heads * num_class, bias=False, weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_c.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        
        N, H, C = x.size(0), self.heads, self.out_channels
        H, Q = self.heads, self.num_class

        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        q= self.lin_c(x).view(-1, H, Q)
        x = self.lin(x).view(-1, H, C)
        self.h = q.clone()
                
        # propagate_type: (x: Tensor)
        out = self.propagate(edge_index, x=x,q=q, size=None)

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor, q_i: Tensor, q_j: Tensor,
                size_i: Optional[int]) -> Tensor:
        alpha = self.get_attention(edge_index_i, q_i, q_j, num_nodes=size_i)
        self.alpha_ = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if self.oracle_attention:
            alpha = self.oracle_alpha_
        return x_j * alpha.unsqueeze(-1)

    def get_attention(self, edge_index_i: Tensor, q_i: Tensor, q_j: Tensor,
                      num_nodes: Optional[int]) -> Tensor:
                              
        if self.class_num == "Single":
            y_pred_i = F.softmax(q_i,dim=-1)
            y_pred_j = F.softmax(q_j,dim=-1)
        elif self.class_num == "Multi":
            y_pred_i = torch.sigmoid(q_i)
            y_pred_j = torch.sigmoid(q_j)
        
        if self.attention_type == "YDP":
            alpha = (y_pred_i * y_pred_j).sum(-1)
        elif self.attention_type =="YSD":
            alpha = (y_pred_i * y_pred_j).sum(-1)/math.sqrt(self.out_channels)
        
        alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)
        return alpha
    
    def get_oracle_attention(self,head,edge_index,y,gpu_id,with_self_loops=True):
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        y = y.squeeze()
        num_nodes = y.size(0)
        # Add self-loops and sort by index
        if with_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E + N]
        
        oracle_attention = torch.Tensor(0).to(device)
        self_loop_oracle_attention = torch.Tensor(0).to(device)

        for node_idx, label in enumerate(y):
            neighbors, _ = edge_index[:, edge_index[1] == node_idx]
            y_neighbors = y[neighbors]
            if len(label.size()) == 0:
                agree_dist = (y_neighbors == label).float()
            else:  # multi-label case
                agree_dist = (y_neighbors * label).float().sum(dim=1)

            agree_dist = agree_dist / agree_dist.sum()
            oracle_attention = torch.cat((oracle_attention,agree_dist[:-1]),dim=0)
            self_loop_oracle_attention = torch.cat((self_loop_oracle_attention,agree_dist[-1:]),dim=0)

        oracle_attention = torch.cat((oracle_attention,self_loop_oracle_attention),dim=0).unsqueeze(dim=-1)
        oracle_attention = oracle_attention.repeat(1,head)
        self.oracle_alpha_ = oracle_attention

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                )

class GATConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 concat: bool = True,dropout: float = 0.0, 
                 add_self_loops: bool = True,
                 bias: bool = True,attention_type: str = 'SD',**kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.alpha_ = None
        self.attention_type=attention_type

        self.lin = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        
        N, H, C = x.size(0), self.heads, self.out_channels

        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        x = self.lin(x).view(-1, H, C)

        # propagate_type: (x: Tensor)
        out = self.propagate(edge_index, x=x, size=None)

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,
                size_i: Optional[int]) -> Tensor:
        alpha = self.get_attention(edge_index_i, x_i, x_j, num_nodes=size_i)
        self.alpha_ = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def get_attention(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,
                      num_nodes: Optional[int]) -> Tensor:                      
        
        if self.attention_type == "DP":
            alpha = (x_i * x_j).sum(-1)
        elif self.attention_type =="SD":
            alpha = (x_i * x_j).sum(-1)/math.sqrt(self.out_channels)
        
        alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)
        return alpha
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                )