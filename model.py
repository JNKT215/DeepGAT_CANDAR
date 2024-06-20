import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
)
from layer import DeepGATConv,GATConv

class DeepGAT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.dropout = cfg['dropout']
        self.cfg = cfg
        self.mid_norms = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.mid_lins = nn.ModuleList()
        
        if cfg['norm'] == 'LayerNorm':
            self.in_norm = nn.LayerNorm(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.LayerNorm(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.LayerNorm(cfg['n_class'])
        elif cfg['norm'] == 'BatchNorm1d':
            self.in_norm = nn.BatchNorm1d(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.BatchNorm1d(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.BatchNorm1d(cfg['n_class'])
        else:
            self.in_norm = nn.Identity()
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.Identity())
            self.out_norm = nn.Identity()
        
        
        if cfg["num_layer"] == 1:
            if cfg['task'] == 'Transductive':
                self.outconv = DeepGATConv(in_channels=cfg['n_feat'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
            elif cfg['task'] == 'Inductive':
                self.outconv = DeepGATConv(cfg['n_feat'], cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.out_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_class'])
        else: 
            if cfg['task'] == 'Transductive':
                self.inconv = DeepGATConv(in_channels=cfg['n_feat'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                for _ in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(DeepGATConv(in_channels=cfg['n_hid']*cfg['n_head'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention']))
                self.outconv = DeepGATConv(in_channels=cfg['n_hid']*cfg['n_head'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
            elif cfg['task'] == 'Inductive':
                self.inconv = DeepGATConv(cfg['n_feat'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.in_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_head'] * cfg['n_hid'])
                for _ in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(DeepGATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention']))
                    self.mid_lins.append(torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_head'] * cfg['n_hid']))
                self.outconv = DeepGATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.out_lin = torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_class'])

    def forward(self, x, edge_index):
        hs = []
        if self.cfg['task'] == 'Transductive':
            if self.cfg["num_layer"] !=1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x= self.inconv(x,edge_index)
                x = self.in_norm(x)
                hs.append(self.inconv.h)
                x = F.elu(x)
            for mid_conv,mid_norm in zip(self.mid_convs,self.mid_norms):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = mid_conv(x, edge_index)
                x = mid_norm(x)
                hs.append(mid_conv.h)
                x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.outconv(x,edge_index)
            x = self.out_norm(x)
            hs.append(self.outconv.h)
        elif self.cfg['task'] == 'Inductive':
            if self.cfg["num_layer"] !=1:
                x = self.inconv(x, edge_index) + self.in_lin(x)
                x = self.in_norm(x)
                hs.append(self.inconv.h)
                x = F.elu(x)
            for mid_conv,mid_lin,mid_norm in zip(self.mid_convs,self.mid_lins,self.mid_norms):
                x = mid_conv(x, edge_index) + mid_lin(x)
                x = mid_norm(x)
                hs.append(mid_conv.h)
                x = F.elu(x)          
            x = self.outconv(x, edge_index) + self.out_lin(x)
            x = self.out_norm(x)
            hs.append(self.outconv.h)
        return x,hs,self.outconv.alpha_
    
    def get_v_attention(self, edge_index,num_nodes,att):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E]

        v_att_l = []
        for v_index in range(num_nodes):
            att_neighbors = att[edge_index[1] == v_index, :].t()  # [heads, #neighbors]
            att_neighbors = att_neighbors.mean(dim=0)
            v_att_l.append(att_neighbors.to('cpu').detach().numpy().copy())

        return v_att_l
    
    def set_oracle_attention(self,edge_index,y,with_self_loops=True):
            if self.cfg["num_layer"] !=1:
                self.inconv.get_oracle_attention(self.cfg['n_head'],edge_index,y,with_self_loops)
            for i in range(self.cfg["num_layer"]-2):
                self.mid_convs[i].get_oracle_attention(self.cfg['n_head'],edge_index,y,with_self_loops)
            self.outconv.get_oracle_attention(self.cfg['n_head_last'],edge_index,y,with_self_loops)

    


class GAT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.dropout = cfg['dropout']
        self.cfg = cfg
        self.mid_norms = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.mid_lins = nn.ModuleList()

        if cfg['norm'] == 'LayerNorm':
            self.in_norm = nn.LayerNorm(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.LayerNorm(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.LayerNorm(cfg['n_class'])
        elif cfg['norm'] == 'BatchNorm1d':
            self.in_norm = nn.BatchNorm1d(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.BatchNorm1d(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.BatchNorm1d(cfg['n_class'])
        else:
            self.in_norm = nn.Identity()
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.Identity())
            self.out_norm = nn.Identity()
        
        if cfg["num_layer"] == 1:
            if cfg['task'] == 'Transductive':
                self.outconv = GATConv(in_channels=cfg['n_feat'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"])
            elif cfg['task'] == 'Inductive':
                self.outconv = GATConv(cfg['n_feat'], cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg["att_type"])
                self.out_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_class'])
        else:
            if cfg['task'] == 'Transductive':
                self.inconv = GATConv(in_channels=cfg['n_feat'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"])
                for _ in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(GATConv(in_channels=cfg['n_hid']*cfg['n_head'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"]))
                self.outconv = GATConv(in_channels=cfg['n_hid']*cfg['n_head'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"])
            elif cfg['task'] == 'Inductive':
                self.inconv = GATConv(cfg['n_feat'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg["att_type"])
                self.in_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_head'] * cfg['n_hid'])
                for _ in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(GATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg["att_type"]))
                    self.mid_lins.append(torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_head'] * cfg['n_hid']))
                self.outconv = GATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg["att_type"])
                self.out_lin = torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_class'])
    
    def forward(self, x, edge_index):
        if self.cfg['task'] == 'Transductive':
            if self.cfg["num_layer"] !=1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x= self.inconv(x,edge_index)
                x = self.in_norm(x)
                x = F.elu(x)
            for mid_conv,mid_norm in zip(self.mid_convs,self.mid_norms):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = mid_conv(x, edge_index)
                x = mid_norm(x)
                x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.outconv(x,edge_index)
            x = self.out_norm(x)
        elif self.cfg['task'] == 'Inductive':
            if self.cfg["num_layer"] !=1:
                x = self.inconv(x, edge_index) + self.in_lin(x)
                x = self.in_norm(x)
                x = F.elu(x)
            for mid_conv,mid_lin,mid_norm in zip(self.mid_convs,self.mid_lins,self.mid_norms):
                x = mid_conv(x, edge_index) + mid_lin(x)
                x = mid_norm(x)
                x = F.elu(x)          
            x = self.outconv(x, edge_index) + self.out_lin(x)
            x = self.out_norm(x)
        return x,[],self.outconv.alpha_

    def get_v_attention(self, edge_index,num_nodes,att):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E]

        v_att_l = []
        for v_index in range(num_nodes):
            att_neighbors = att[edge_index[1] == v_index, :].t()  # [heads, #neighbors]
            att_neighbors = att_neighbors.mean(dim=0)
            v_att_l.append(att_neighbors.to('cpu').detach().numpy().copy())

        return v_att_l