import sys
sys.path.append("/home/mpc/LMILP/LAMPOS/")
from predictions import  bin_pred
from predictions import bin_pred_FC
import gzip
import _pickle as pickle
import torch.nn.functional as Fun
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
from utils import get_recentdate_str

# Select the horizon and sampling time

N=40
dt=0.1

# Pick the most recent processed files with the selected horizon and sampling time

dataset_path = "Offline/Datasets"
recent_date_str,dt_str=get_recentdate_str(dataset_path=dataset_path,N=N,dt=dt)

# Dataset and labels dictionary loading

PIK="Offline/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
PIK1="Offline/ProcessedFiles/bin_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
PIK2="Offline/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"

with gzip.open(PIK, "rb") as f:
    p=pickle.Unpickler(f)
    dataset=p.load()
with gzip.open(PIK1, "rb") as f:
    p=pickle.Unpickler(f)
    bin_labels=p.load()           
with gzip.open(PIK2, "rb") as f:
    p=pickle.Unpickler(f)
    bin_map=p.load()              

# Dataset related parameter

binary_dim=len(list(bin_map.values())[0])
state_dim=len(dataset[0][0])
binary_classes_dim=len(bin_labels)
bin_map_updated={}
bin_labels_updated={}
bin_labels_new={}
classes_labels=list(bin_labels.keys()) 

# Training parameters

num_epochs=len(dataset)*32
batch_size=256

# Network creation

device = torch.device("cpu")
bin_pred_model=bin_pred_FC(input_size=state_dim, output_size=binary_classes_dim, hidden_size=512,depth=3,device=device)
bin_pred_model=bin_pred_model.to(device=device)
bin_pred_model_path="Offline/TrainedModels/bin_pred_model_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"

load=False
if  load:
    bin_pred_model.load_state_dict(torch.load(bin_pred_model_path,map_location=torch.device('cpu')))

# Loss Function and optimizer definition

criterion= torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(bin_pred_model.parameters(), lr=1e-2)

# Training

num_data=len(dataset)
medium_loss=0
classes_batches=[]
for cl in classes_labels:
    data_set_cl=[dataset[i] for i in bin_labels[cl]]
    classes_batches.append(data_set_cl)
for  epoch in range(num_epochs+1) :
  
    batch=[]
    for cl in classes_batches:
        batch_cl=random.sample(cl,int(batch_size/8))
        batch.extend(batch_cl)
    input=[torch.tensor((np.array(batch[i][0]).astype(np.float32))) for i in range(len(batch))]
    input=torch.stack(input).to(device)
    bin_out=torch.stack([torch.tensor(np.array(classes_labels.index(batch[i][2]))) for i in range(len(batch))]).to(device)#if batch[i][2]!=482
    y_pred = bin_pred_model.forward(input) 
    loss = criterion(y_pred,bin_out)

    # Obtaining the loss function

    optimizer.zero_grad()
    loss.backward()         
    medium_loss=medium_loss+loss.item()
    grad_norm=0.
    for p in bin_pred_model.parameters():
        grad_norm=max(grad_norm, p.grad.detach().data.norm(2))
    optimizer.step() 
    #Print Loss and Gradient
    if  epoch % 100 == 0 and epoch>0:
        print("Epoch: %d, loss: %1.5f,  grad: %1.5f" % (epoch, medium_loss/100, grad_norm))
        if medium_loss/100<.01 :
            break
        medium_loss=0 

# Saving the network parameters

torch.save(bin_pred_model.state_dict(),bin_pred_model_path)
     


   