import sys
sys.path.append("/home/mpc/LMILP/LAMPOS/")
from predictions import  cov_pred
import gzip
import _pickle as pickle
import torch.nn.functional as Fun
import torch 
import torch.nn as nn
from utils import get_recentdate_str
import random
import numpy as np


# Select the horizon and sampling time

N=40
dt=0.1

#Pick the most recent files with the selected horizon and sampling time

dataset_path = "Offline/Datasets"
recent_date_str,dt_str=get_recentdate_str(dataset_path=dataset_path,N=N,dt=dt)

# Dataset and labels dictionary loading 

PIK="Offline/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
PIK1="Offline/ProcessedFiles/cov_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
PIK2="Offline/ProcessedFiles/cov_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"

cov_map={}
with gzip.open(PIK, "rb") as f:
    p=pickle.Unpickler(f)
    data_set_pre=p.load()
with gzip.open(PIK1, "rb") as f1:
    while True:
        try:
            p=pickle.Unpickler(f1)
            cov_list_k=p.load()
            cov_map.update({cov_list_k[0]:cov_list_k[1]})
        except EOFError:
            break
with gzip.open(PIK2, "rb") as f:
    p=pickle.Unpickler(f)
    cov_labels=p.load()

#Dataset related parameters

input_size = len(data_set_pre[0][0]) 
num_cov_classes=len(cov_map)
cover_classes_dim=num_cov_classes
classes_labels=list(cov_map.keys())

# Training parameters

num_epochs=len(data_set_pre)*100
hidden_size = 128
learning_rate=1e-2
depth=3

#Network creation

device = torch.device("cpu")
cov_pred_model_path="Offline/TrainedModels/cov_pred_model_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
cov_pred_model=cov_pred(input_size,hidden_size,cover_classes_dim,depth,device)
load=False
if  load:
    cov_pred_model.load_state_dict(torch.load(cov_pred_model_path,map_location=torch.device('cpu')))

#Loss function and optimizer definition

criterion = torch.nn.CrossEntropyLoss()    # mean-squared error for regression
cov_pred_model=cov_pred_model.to(device=device)
optimizer = torch.optim.SGD(cov_pred_model.parameters(), lr=1e-2)

#Training

num_data=len(data_set_pre)
medium_loss=0
batchsize=16#256
for epoch in range(num_epochs+1):
    batch=[]
    for cl in classes_labels:
        data_set_pre_cl=[data_set_pre[i] for i in cov_labels[cl]]
        batch_cl=random.sample(data_set_pre_cl,int(batchsize/16))
        batch.extend(batch_cl)
    batch_input=torch.tensor(np.array([np.array(batch[i][0]).astype(np.float32) for i in range(len(batch))]),device=device)
    batch_cov=torch.tensor(np.array([batch[i][1] for i in range(len(batch))]),device=device)
    cov_out = cov_pred_model.forward(batch_input) 
    loss = criterion(cov_out, batch_cov)
    optimizer.zero_grad() 
    loss.backward()        
    medium_loss=medium_loss+loss.item()
    grad_norm=0.
    for p in cov_pred_model.parameters():
        grad_norm=max(grad_norm, p.grad.detach().data.norm(2))
    optimizer.step() 
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f,  grad: %1.5f" % (epoch, medium_loss/100, grad_norm))
        medium_loss=0 

#Saving the network parameters

torch.save(cov_pred_model.state_dict(),cov_pred_model_path)
