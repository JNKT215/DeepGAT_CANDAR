import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_dense_adj,
)
from layer import DeepGATConv,GATConv

class DeepGAT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.dropout = cfg['dropout']
        self.cfg = cfg
        self.n_feat = cfg['n_feat'] +cfg['n_class']  if self.cfg['label_feat'] else cfg['n_feat']
        self.n_hid_list = self.get_n_hid_list()
        self.n_hid_multiple_n_head_list = [(n_hid+ cfg['n_class']) * cfg['n_head']  for n_hid in self.n_hid_list] if self.cfg['label_feat'] else [n_hid * cfg['n_head'] for n_hid in self.n_hid_list]
        self.rm_diag_row_normalized_adjs = None
        self.mid_norms = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.mid_lins = nn.ModuleList()
        
        if cfg['norm'] == 'LayerNorm':
            self.in_norm = nn.LayerNorm(self.n_hid_list[0] * cfg['n_head'])
            for n_layer in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.LayerNorm(self.n_hid_list[n_layer] * cfg['n_head']))
            self.out_norm = nn.LayerNorm(cfg['n_class'])
        elif cfg['norm'] == 'BatchNorm1d':
            self.in_norm = nn.BatchNorm1d(self.n_hid_list[0] * cfg['n_head'])
            for n_layer in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.BatchNorm1d(self.n_hid_list[n_layer] * cfg['n_head']))
            self.out_norm = nn.BatchNorm1d(cfg['n_class'])
        else:
            self.in_norm = nn.Identity()
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.Identity())
            self.out_norm = nn.Identity()
        
        
        if cfg["num_layer"] == 1:
            if cfg['task'] == 'Transductive':
                self.outconv = DeepGATConv(in_channels=self.n_feat, out_channels=cfg['n_class'],num_class=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
            elif cfg['task'] == 'Inductive':
                self.outconv = DeepGATConv(in_channels=self.n_feat, out_channels=cfg['n_class'],num_class=cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.out_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_class'])
        else: 
            if cfg['task'] == 'Transductive':
                self.inconv = DeepGATConv(in_channels=self.n_feat,out_channels=cfg['n_hid'],num_class=cfg['n_class'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                for n_layer in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(DeepGATConv(in_channels=self.n_hid_multiple_n_head_list[n_layer-1],out_channels=self.n_hid_list[n_layer],num_class=cfg['n_class'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention']))
                self.outconv = DeepGATConv(in_channels=self.n_hid_multiple_n_head_list[-1], out_channels=cfg['n_class'],num_class=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
            elif cfg['task'] == 'Inductive':
                self.inconv = DeepGATConv(in_channels=self.n_feat, out_channels=cfg['n_hid'],num_class=cfg['n_class'], heads=cfg['n_head'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.in_lin = torch.nn.Linear(cfg['n_feat'], self.n_hid_multiple_n_head_list[0])
                for n_layer in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(DeepGATConv(in_channels=self.n_hid_multiple_n_head_list[n_layer-1], out_channels=self.n_hid_list[n_layer],num_class=cfg['n_class'], heads=cfg['n_head'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention']))
                    self.mid_lins.append(torch.nn.Linear(self.n_hid_multiple_n_head_list[n_layer], self.n_hid_multiple_n_head_list[n_layer]))
                self.outconv = DeepGATConv(in_channels=self.n_hid_multiple_n_head_list[-1], out_channels=cfg['n_class'],num_class=cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.out_lin = torch.nn.Linear(self.n_hid_multiple_n_head_list[-1], cfg['n_class'])

    def forward(self, x, edge_index,y_feat=None):
        if self.cfg['label_feat']: y_feats = [rm_diag_adj @ y_feat for rm_diag_adj in self.rm_diag_row_normalized_adjs]
        hs = []
        if self.cfg['task'] == 'Transductive':
            if self.cfg["num_layer"] !=1:
                if self.cfg['label_feat']: x = self.cat_x_and_y_feat(x=x,y_feats=y_feats,n_layer=0)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x= self.inconv(x,edge_index)
                x = self.in_norm(x)
                hs.append(self.inconv.h)
                x = F.elu(x)
            for n_layer,(mid_conv,mid_norm) in enumerate(zip(self.mid_convs,self.mid_norms),1):
                if self.cfg['label_feat']: x = self.cat_x_and_y_feat(x=x,y_feats=y_feats,n_layer=n_layer)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = mid_conv(x, edge_index)
                x = mid_norm(x)
                hs.append(mid_conv.h)
                x = F.elu(x)
            if self.cfg['label_feat']: x = self.cat_x_and_y_feat(x=x,y_feats=y_feats,n_layer=self.cfg['num_layer']-1) # L-layer
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.outconv(x,edge_index)
            x = self.out_norm(x)
            hs.append(self.outconv.h)
        elif self.cfg['task'] == 'Inductive':
            if self.cfg["num_layer"] !=1:
                if self.cfg['label_feat']: self.cat_x_and_y_feat(x=x,y_feats=y_feats,n_layer=0)
                x = self.inconv(x, edge_index) + self.in_lin(x)
                x = self.in_norm(x)
                hs.append(self.inconv.h)
                x = F.elu(x)
            for n_layer,(mid_conv,mid_lin,mid_norm) in enumerate(zip(self.mid_convs,self.mid_lins,self.mid_norms),1):
                if self.cfg['label_feat']: x = self.cat_x_and_y_feat(x=x,y_feats=y_feats,n_layer=n_layer)
                x = mid_conv(x, edge_index) + mid_lin(x)
                x = mid_norm(x)
                hs.append(mid_conv.h)
                x = F.elu(x)          
            if self.cfg['label_feat']: x = self.cat_x_and_y_feat(x=x,y_feats=y_feats,n_layer=self.cfg['num_layer']-1) # L-layer
            x = self.outconv(x, edge_index) + self.out_lin(x)
            x = self.out_norm(x)
            hs.append(self.outconv.h)
        return x,hs,self.outconv.alpha_
    
    def dim_reduction_per_l(self,n_layer):
        return int(self.cfg['n_hid'] - ((self.cfg['n_hid'] - self.cfg["n_class"]) / (self.cfg["num_layer"] -2) * n_layer))
    
    def cat_x_and_y_feat(self,x,y_feats,n_layer):
        if n_layer==0:
            return torch.cat((x,y_feats[n_layer]),dim=-1)            
        x = x.view(-1,self.cfg["n_head"],self.n_hid_list[n_layer-1])
        y_feat = y_feats[n_layer].unsqueeze(1).repeat(1,self.cfg['n_head'],1)
        x = torch.cat((x,y_feat),dim=-1)
        x = x.view(-1,self.cfg['n_head']*(self.n_hid_list[n_layer-1]+self.cfg['n_class']))
        return x
        
    
    def get_n_hid_list(self):
        n_hid_list = [self.cfg['n_hid']]        
        for n_layer in range(1,self.cfg["num_layer"]-1):
            n_hid = self.dim_reduction_per_l(n_layer=n_layer)
            n_hid_list.append(n_hid)

        return n_hid_list

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
            
    def set_l_hops_rm_diag_row_normalized_adj(self,edge_index,num_nodes,with_self_loops=True):
        def rm_diag(matrix):
            matrix = matrix.clone()
            diag_indices = torch.arange(matrix.size(0))
            matrix[diag_indices, diag_indices] = 0
            return matrix
        def get_row_normalize_adj(adj):
            row_sum = adj.sum(dim=1) 
            row_sum[row_sum == 0] = 1  # avoid division by zero
            row_normalized_adj = adj / row_sum.view(-1, 1)
            return row_normalized_adj
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Add self-loops and sort by index
        if with_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E + N]
        
        # calc adj_matrix
        adj = torch.zeros(num_nodes,num_nodes).to(device)
        for i in range(len(edge_index[0])):
            adj[edge_index[0][i]][edge_index[1][i]] = 1
        adj = adj.float()
        
        #get row normarized adj
        row_normalized_adj = get_row_normalize_adj(adj)
    
        row_normalized_adjs = [row_normalized_adj ** (n_layer) for n_layer in range(1,int(self.cfg['num_layer']))]
        row_normalized_adjs.insert(0,torch.eye(num_nodes).float().to(device))
        
        self.rm_diag_row_normalized_adjs = [rm_diag(adj) for adj in row_normalized_adjs]
        
    


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


class Loss_func(nn.Module):
    def __init__(self,cfg): # パラメータの設定など初期化処理を行う
        super().__init__()
        self.cfg = cfg
        self.loss_op = torch.nn.BCEWithLogitsLoss()

    def forward(self,out,y,mask=None,hs=[]): # モデルの出力と正解データ
        if self.cfg['task'] == 'Transductive':
            if self.cfg['model'] == "GAT":
                out_softmax = F.log_softmax(out,dim=-1)
                loss = F.nll_loss(out_softmax[mask],y[mask])
            elif self.cfg['model'] == "DeepGAT":
                out_softmax = F.log_softmax(out,dim=-1)
                loss = F.nll_loss(out_softmax[mask],y[mask]) * self.ganma_l(num_layer=self.cfg['num_layer'])
                loss += self.get_y_preds_loss(hs,y,mask)
                
        elif self.cfg['task'] == 'Inductive':
            if self.cfg['model'] == "GAT":
                loss = self.loss_op(out,y)
            if self.cfg['model'] == "DeepGAT":
                loss = self.loss_op(out,y) * self.ganma_l(num_layer=self.cfg['num_layer'])
                loss += self.get_y_preds_loss_ppi(hs,y)
                
        return loss
    
    def ganma_l(self,num_layer): 
        return self.cfg['delta'] * (num_layer + self.cfg['delta'])**(-1) +1
        # return 1 * (self.cfg['delta'] +1) **((num_layer)*-(1)) +1
    
    def get_y_preds_loss(self,hs,y,mask):
        y_pred_loss = torch.tensor(0, dtype=torch.float32,device=hs[0].device)
        
        for n_layer,h in enumerate(hs):
            h = h.mean(dim=1)
            y_pred = F.log_softmax(h, dim=-1)
            y_pred_loss += F.nll_loss(y_pred[mask],y[mask]) * self.ganma_l(num_layer=n_layer)

        return y_pred_loss

    def get_y_preds_loss_ppi(self,hs,y):
        y_pred_loss = torch.tensor(0, dtype=torch.float32,device=hs[0].device)
        
        for n_layer,h in enumerate(hs):
            h = h.mean(dim=1)
            y_pred_loss += self.loss_op(h,y) * self.ganma_l(num_layer=n_layer)
            
        return y_pred_loss
    
    
def return_model(cfg,data=None):
    if cfg['model'] == 'GAT':
            model = GAT(cfg)
    elif cfg['model'] == 'DeepGAT':
        model = DeepGAT(cfg)
        if cfg['label_feat']:
            model.set_l_hops_rm_diag_row_normalized_adj(data.edge_index,data.num_nodes)
        if cfg['oracle_attention']:
            model.set_oracle_attention(data.edge_index,data.y)
    
    return model