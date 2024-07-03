import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from model import return_model,Loss_func
import hydra
from hydra import utils
from tqdm import tqdm
import mlflow
from utils import EarlyStopping,set_seed,log_artifacts,accuracy,set_label_features,intermediate_result_print

def train(data, model, optimizer):
    model.train()
    loss_func = Loss_func(cfg=model.cfg)
    optimizer.zero_grad()
    if model.cfg["label_feat"]:
        out_train,hs,_ = model(data.x, data.edge_index, data.y_feat)
    else:
        out_train,hs,_ = model(data.x, data.edge_index)
    loss_train = loss_func(out_train,data.y,data.train_mask,hs)
    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    if model.cfg["label_feat"]:
        out_val,_,_ = model(data.x, data.edge_index, data.y_feat)
    else:
        out_val,_,_ = model(data.x, data.edge_index)
    out_val_softmax = F.log_softmax(out_val, dim=-1)
    loss_val = F.nll_loss(out_val_softmax[data.val_mask], data.y[data.val_mask])

    return loss_val.item()


@torch.no_grad()
def test(data,model):
    model.eval()
    if model.cfg["label_feat"]:
        out,_,attention = model(data.x, data.edge_index, data.y_feat)
    else:
        out,_,attention = model(data.x, data.edge_index)
    out_softmax = F.log_softmax(out, dim=1)
    acc = accuracy(out_softmax,data,'test_mask')
    attention = model.get_v_attention(data.edge_index,data.x.size(0),attention)
    return acc,attention,out

def run(data,model,optimizer,cfg):

    early_stopping = EarlyStopping(cfg['patience'],path=cfg['path'])

    for epoch in range(cfg['epochs']):
        loss_val = train(data,model,optimizer)
        intermediate_result_print(dataset=cfg['dataset'],epoch=epoch,data=data,model=model,test=test)
        if early_stopping(loss_val,model,epoch) is True:
            break
    
    model.load_state_dict(torch.load(cfg['path']))
    test_acc,attention,h = test(data,model)
    print(f"dataset:{cfg['dataset']}, best epoch{early_stopping.epoch}, test_acc:{test_acc* 100}")
    return test_acc * 100,early_stopping.epoch,attention,h


@hydra.main(config_path='conf', config_name='config')
def main(cfg):

    print(utils.get_original_cwd())
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment(cfg.experiment_name)
    mlflow.start_run()
    
    cfg = cfg[cfg.key]

    for key,value in cfg.items():
        mlflow.log_param(key,value)
        
    root = utils.get_original_cwd() + '/data/' + cfg['dataset']
    dataset = Flickr(root= root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    if cfg['label_feat']:
        data.y_feat = set_label_features(y=data.y,num_nodes=data.num_nodes,num_class=cfg['n_class'],dataset=cfg['dataset'],train_mask=data.train_mask,device=device)
    train_index, val_index = torch.nonzero(data.train_mask).squeeze(),torch.nonzero(data.val_mask).squeeze()
    
    
    artifacts,test_accs,epochs,attentions,hs = {},[],[],[],[]
    artifacts[f"{cfg['dataset']}_y_true.npy"] = data.y
    artifacts[f"{cfg['dataset']}_x.npy"] = data.x
    artifacts[f"{cfg['dataset']}_supervised_index.npy"] = torch.cat((train_index,val_index),dim=0)
    for i in tqdm(range(cfg['run'])):
        set_seed(i)
        model = return_model(cfg,data).to(device)
            
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg["learing_late"],weight_decay=cfg['weight_decay'])
        test_acc,epoch,attention,h = run(data,model,optimizer,cfg)
        
        test_accs.append(test_acc)
        epochs.append(epoch)
        attentions.append(attention)
        hs.append(h)
        del model
        torch.cuda.empty_cache()
        
    acc_max_index = test_accs.index(max(test_accs))
    artifacts[f"{cfg['dataset']}_{cfg['att_type']}_attention_L{cfg['num_layer']}.npy"] = attentions[acc_max_index]
    artifacts[f"{cfg['dataset']}_{cfg['att_type']}_h_L{cfg['num_layer']}.npy"] = hs[acc_max_index]
            
    test_acc_ave = sum(test_accs)/len(test_accs)
    epoch_ave = sum(epochs)/len(epochs)
    log_artifacts(artifacts,output_path=f"{utils.get_original_cwd()}/DeepGAT/output/{cfg['dataset']}/{cfg['att_type']}/oracle/{cfg['oracle_attention']}")
        
    mlflow.log_metric('epoch_mean',epoch_ave)
    mlflow.log_metric('test_acc_min',min(test_accs))
    mlflow.log_metric('test_acc_mean',test_acc_ave)
    mlflow.log_metric('test_acc_max',max(test_accs))
    mlflow.end_run()
    return test_acc_ave

    
    

if __name__ == "__main__":
    main()