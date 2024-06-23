import torch
import numpy as np
import os
import mlflow
import torch.nn.functional as F

class EarlyStopping():
    def __init__(self,patience,path="checkpoint.pt"):
        self.best_loss_score = None
        self.loss_counter =0
        self.patience = patience
        self.path = path
        self.val_loss_min =None
        self.epoch = 0
        
    def __call__(self,loss_val,model,epoch):
        if self.best_loss_score is None:
            self.best_loss_score = loss_val
            self.save_best_model(model,loss_val)
            self.epoch = epoch
        elif self.best_loss_score > loss_val:
            self.best_loss_score = loss_val
            self.loss_counter = 0
            self.save_best_model(model,loss_val)
            self.epoch = epoch
        else:
            self.loss_counter+=1
            
        if self.loss_counter == self.patience:
            return True
        
        return False
    def save_best_model(self,model,loss_val):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = loss_val

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_train_label_per(data):
    cnt = 0
    for i in data.train_mask:
        if i == True:
            cnt+=1

    train_mask_label = cnt
    labels_num = len(data.train_mask)
    train_label_percent = train_mask_label/labels_num

    print(f"train_mask_label:{cnt},labels_num:{labels_num},train_label_percent:{train_label_percent}")

def log_artifacts(artifacts,output_path=None):
    if artifacts is not None:
        for artifact_name, artifact in artifacts.items():
            if isinstance(artifact, list):
                if output_path is not None:
                    artifact_name = f"{output_path}/{artifact_name}"
                    os.makedirs(output_path, exist_ok=True)
                np.save(artifact_name, artifact)
                mlflow.log_artifact(artifact_name)
            elif artifact is not None and artifact !=[]:
                if output_path is not None:
                    artifact_name = f"{output_path}/{artifact_name}"
                    os.makedirs(output_path, exist_ok=True)
                np.save(artifact_name, artifact.to('cpu').detach().numpy().copy())
                mlflow.log_artifact(artifact_name)
                
def accuracy(out,data,mask):
    mask = data[mask]
    acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
    return acc

def set_label_features(y,num_nodes,num_class,dataset,train_mask,device):
    if dataset != "PPI": #dataset == (CS or Physics or Flickr)
        onthot_label_features = torch.zeros((num_nodes, num_class)).to(device)
        onthot_label_features[train_mask] = F.one_hot(y[train_mask],num_class).float()
        # onthot_label_features = F.one_hot(y, num_class).float()
    else: #dataset == PPI
        onthot_label_features = torch.zeros((num_nodes, num_class))
        # onthot_label_features[train_mask] = y[train_mask].float()
        onthot_label_features = y.float()
    
    return onthot_label_features

def intermediate_result_print(dataset,epoch,data,model,test):
    if dataset != "PPI": #dataset == (CS or Physics or Flickr)
        _,_,out = test(data,model)
        out_softmax = F.log_softmax(out, dim=1)
        train_acc = accuracy(out_softmax,data,'train_mask')
        val_acc   = accuracy(out_softmax,data,'val_mask')
        test_acc  = accuracy(out_softmax,data,'test_mask')
    else: #dataset==PPI
        pass
        
    print(f"dataset:{dataset}, current epoch:{epoch}\n train_acc:{train_acc*100:.4f}, val_acc:{val_acc*100:.4f}, test_acc{test_acc*100:.4f}")