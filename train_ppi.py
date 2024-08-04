import torch
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from model import return_model,Loss_func
import hydra
from hydra import utils
from tqdm import tqdm
import mlflow
from utils import EarlyStopping,set_seed,log_artifacts

@torch.no_grad()
def get_train_h(train_loader,model,device):
    model.eval()
    train_hs = []
    for data in train_loader:  # in [g1, g2, ..., g20]
        out,_,_ = model(data.x.to(device), data.edge_index.to(device))
        train_hs.append(out)
    return torch.cat(train_hs, dim=0)

def train(loader,model,optimizer,device):
    model.train()
    loss_func = Loss_func(cfg=model.cfg)
    total_loss = 0
                
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out,hs,_ = model(data.x, data.edge_index)
        loss = loss_func(out=out,y=data.y,hs=hs)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
            
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(loader,model,device):
    model.eval()
    ys, preds,attentions,hs = [], [], [], []
    for data in loader: # only one batch (=g1+g2)
        ys.append(data.y)
        out,_,attention = model(data.x.to(device), data.edge_index.to(device))
        attention = model.get_v_attention(data.edge_index,data.x.size(0),attention)
        attentions.append(attention)
        hs.append(out)
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0, attentions[0],hs[0]

def run(loader,model,optimizer,device,cfg):

    train_loader,test_loader = loader
    early_stopping = EarlyStopping(cfg['patience'],path=cfg['path'])

    for epoch in range(cfg['epochs']):
        loss_val = train(train_loader,model,optimizer,device)
        if early_stopping(loss_val,model,epoch) is True:
            break
    
    model.load_state_dict(torch.load(cfg['path']))
    test_acc,attention,test_h = test(test_loader,model,device)
    train_h = get_train_h(train_loader,model,device)
    return test_acc * 100,early_stopping.epoch,attention,test_h,train_h

@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    torch.cuda.empty_cache()
    print(utils.get_original_cwd())
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment(cfg.experiment_name)
    mlflow.start_run()

    cfg = cfg[cfg.key]

    for key,value in cfg.items():
        mlflow.log_param(key,value)
    
    root = utils.get_original_cwd() + '/data/' + cfg['dataset']
    train_dataset = PPI(root, split='train')
    val_dataset = PPI(root, split='val')
    test_dataset = PPI(root, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    loader =[train_loader,test_loader]

    device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
    train_xs,train_ys,test_xs,test_ys = [data.x for data in train_loader],[data.y for data in train_loader],[data.x for data in test_loader],[data.y for data in test_loader]
    
    artifacts,test_accs,epochs,attentions,test_hs,train_hs = {},[],[],[],[],[]
    artifacts[f"{cfg['dataset']}_train_y_true.npy"] = torch.cat(train_ys, dim=0)
    artifacts[f"{cfg['dataset']}_train_x.npy"] = torch.cat(train_xs, dim=0)    
    artifacts[f"{cfg['dataset']}_test_y_true.npy"] = test_ys[0]
    artifacts[f"{cfg['dataset']}_test_x.npy"] = test_xs[0] 
    for i in tqdm(range(cfg['run'])):
        set_seed(i)
        model = return_model(cfg).to(device)
             
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learing_late'])
        test_acc,epoch,attention,test_h,train_h = run(loader,model,optimizer,device,cfg)

        test_accs.append(test_acc)
        epochs.append(epoch)
        attentions.append(attention)
        test_hs.append(test_h)
        train_hs.append(train_h)
        del model
        torch.cuda.empty_cache()

    acc_max_index = test_accs.index(max(test_accs))
    artifacts[f"{cfg['dataset']}_{cfg['att_type']}_attention_L{cfg['num_layer']}.npy"] = attentions[acc_max_index]
    artifacts[f"{cfg['dataset']}_{cfg['att_type']}_test_h_L{cfg['num_layer']}.npy"] = test_hs[acc_max_index]
    artifacts[f"{cfg['dataset']}_{cfg['att_type']}_train_h_L{cfg['num_layer']}.npy"] = train_hs[acc_max_index]

    test_acc_ave = sum(test_accs)/len(test_accs)
    epoch_ave = sum(epochs)/len(epochs)
    log_artifacts(artifacts,output_path=f"{utils.get_original_cwd()}/DeepGAT/output/{cfg['dataset']}/{cfg['att_type']}")

    mlflow.log_metric('epoch_mean',epoch_ave)
    mlflow.log_metric('test_acc_min',min(test_accs))
    mlflow.log_metric('test_acc_mean',test_acc_ave)
    mlflow.log_metric('test_acc_max',max(test_accs))
    
    mlflow.end_run()
    return test_acc_ave


    
if __name__ == "__main__":
    main()