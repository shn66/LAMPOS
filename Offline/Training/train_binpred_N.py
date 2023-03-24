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


#Pick the most recent processed files with the selected horizon and sampling time


dataset_path = "Offline/Datasets"
recent_date_str,dt_str=get_recentdate_str(dataset_path=dataset_path,N=N,dt=dt)


# Dataset and labels dictionary loading

PIK="Offline/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"
PIK1="Offline/ProcessedFiles/bin_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"
PIK2="Offline/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with gzip.open(PIK, "rb") as f:
    p=pickle.Unpickler(f)
    dataset=p.load()
with gzip.open(PIK1, "rb") as f:
    p=pickle.Unpickler(f)
    bin_labels_N=p.load()          
with gzip.open(PIK2, "rb") as f:
    p=pickle.Unpickler(f)
    bin_map_N=p.load()              


# Defining the paths in which load/save the trained model

bin_pred_model_paths=[]
for i in range(N):
    bin_pred_model_paths.append("Offline/TrainedModels/bin_pred_model_"+str(i)+"_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p")

# Dataset related parameters

binary_dim=len(list(bin_map_N[0].values())[0])*N
state_dim=len(dataset[0][0])
binary_classes_dim_N=[len(bin_map_N[n]) for n in range(N)]

#Training parameters

classes_labels_N=[list(bin_labels_N[n].keys()) for n in range(N)]#Classification
num_epochs=len(dataset)*3
batch_size=256

# Network creation

load=False
bin_pred_models=[bin_pred_FC(input_size=state_dim, output_size=binary_classes_dim_N[i], hidden_size=64,depth=2,device=device).to(device=device) for i in range(N)]
if  load:
    [bin_pred_models[i].load_state_dict(torch.load(bin_pred_model_paths[i],map_location=device)) for i in range(N)]
criterion= torch.nn.CrossEntropyLoss() 
optimizer = [torch.optim.SGD(bin_pred_models[i].parameters(), lr=1e-2) for i in range(N)]

# Training

num_data=len(dataset)
medium_loss=0
classes_batches_N=[]
for n in range(N):
    classes_batches=[]
    for cl in classes_labels_N[n]:
        data_set_cl=[dataset[i] for i in bin_labels_N[n][cl]]
        classes_batches.append(data_set_cl)
    classes_batches_N.append(classes_batches)

for  epoch in range(num_epochs+1) :

    # Ehnanced bach sampling strategy (this operation assume that in each class there are enough datapoints)
    loss=[]
    grad_norm=0.
    for n in range(N):
        batch=[]
        for cl in classes_batches_N[n]:
            batch_cl=random.sample(cl,int(batch_size/16))
            batch.extend(batch_cl)
        input=torch.stack([torch.tensor((np.array(batch[i][0]).astype(np.float32))) for i in range(len(batch))]).to(device)
        bin_out=torch.stack([torch.tensor(np.array(batch[i][2][n])) for i in range(len(batch))]).to(device)
        y_pred = bin_pred_models[n].forward(input) 
        loss.append(criterion(y_pred,bin_out))
        optimizer[n].zero_grad()
        loss[n].backward()
        medium_loss=medium_loss+(loss[n].item())/N
        
        for p in bin_pred_models[n].parameters():
            grad_norm=max(grad_norm, p.grad.detach().data.norm(2))
        optimizer[n].step()
    if  epoch % 100 == 0 and epoch>0:
        print("Epoch: %d, loss: %1.5f,  grad: %1.5f" % (epoch, medium_loss/100, grad_norm))
        if medium_loss/100<.01 :
            break
        medium_loss=0 

# Saving the network parameters

for i in range(N):
    i_str=str(i)
    torch.save(bin_pred_models[i].state_dict(),bin_pred_model_paths[i])

     


   