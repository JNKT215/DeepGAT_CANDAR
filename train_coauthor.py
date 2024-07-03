import torch
import torch.nn.functional as F
from torch_geometric.datasets import Coauthor
from model import return_model,Loss_func
import torch_geometric.transforms as T
import hydra
from hydra import utils
from tqdm import tqdm
import mlflow
from utils import EarlyStopping,set_seed,check_train_label_per,log_artifacts,accuracy,set_label_features,intermediate_result_print

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_splits(data, num_classes, lcc_mask):
    # Set random splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
    torch.manual_seed(42)
    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data,torch.cat((train_index,val_index),dim=0)

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
    if model.cfg['label_feat']:
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

    dataset = Coauthor(root           = root,
                        name          = cfg['dataset'],
                        transform     = eval(cfg['transform']),
                        pre_transform = eval(cfg['pre_transform']))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    data,index = random_splits(data=data,num_classes=cfg["n_class"],lcc_mask=None)
    if cfg['label_feat']:
        data.y_feat = set_label_features(y=data.y,num_nodes=data.num_nodes,num_class=cfg['n_class'],dataset=cfg['dataset'],train_mask=data.train_mask,device=device)
    # check_train_label_per(data)
    
    artifacts,test_accs,epochs,attentions,hs = {},[],[],[],[]
    artifacts[f"{cfg['dataset']}_y_true.npy"] = data.y
    artifacts[f"{cfg['dataset']}_x.npy"] = data.x
    artifacts[f"{cfg['dataset']}_supervised_index.npy"] = index
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