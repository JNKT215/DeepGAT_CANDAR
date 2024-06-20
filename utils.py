import torch
import numpy as np
import os
import mlflow

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