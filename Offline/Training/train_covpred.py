from predictions import  cov_pred
import gzip
import _pickle as pickle
import torch.nn.functional as Fun
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
import re
import os
import datetime
# TAKING THE MOST RECENT FILE
N=20
dt=0.1
integer=int(0.1)
decimal=int(dt/0.1)
dt_str=str(integer)+str(decimal)
directory = "/home/mpc/LMILP/Datasets"
files = os.listdir(directory)
dates = []

for file in files:
    match = re.search("MILP_data_points_dt"+dt_str+"_N"+str(N)+"_(.*).p", file)
    #match = re.search("MILP_data_points_dt01_N20_(.*).p", file)
    if match:
        date_str = match.group(1)
        date = datetime.datetime.strptime(date_str,"%Y%m%d-%H%M%S")
        dates.append(date)
dates.sort(reverse=True)
recent_date_str=dates[0].strftime("%Y%m%d-%H%M%S")

#PIK="ProcessedFiles/data_set_pr_dt01_N20_"+recent_date_str+".p"
PIK="/home/mpc/LMILP/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_2.p"
#PIK1="ProcessedFiles/cov_map_pr_dt01_N20_"+recent_date_str+".p"
PIK1="/home/mpc/LMILP/ProcessedFiles/cov_map_clean_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_2.p"
#PIK2="ProcessedFiles/cov_labels_pr_dt01_N20_"+recent_date_str+".p"
PIK2="/home/mpc/LMILP/ProcessedFiles/cov_labels_clean_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_2.p"


device = torch.device("cpu")
with gzip.open(PIK, "rb") as f:
    p=pickle.Unpickler(f)
    data_set_pre=p.load()
with gzip.open(PIK1, "rb") as f:
    p=pickle.Unpickler(f)
    cov_map=p.load()
with gzip.open(PIK2, "rb") as f:
    p=pickle.Unpickler(f)
    cov_labels=p.load()

#Parameters determined by the shape of the data
input_size = len(data_set_pre[0][0]) #number of features
num_cov_classes=len(cov_map)
cover_classes_dim=num_cov_classes
classes_labels=list(cov_map.keys())
# Training parameters
num_epochs=len(data_set_pre)*100
hidden_size = 128
learning_rate=1e-2
depth=3
load=False
#Network creation
cov_pred_model_path="/home/mpc/LMILP/TrainedModels/cov_pred_model_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
cov_pred_1=cov_pred(input_size,hidden_size,cover_classes_dim,depth,device)
if  load:
    cov_pred_1.load_state_dict(torch.load(cov_pred_model_path,map_location=torch.device('cpu')))
#Loss function and optimizer definition
criterion = torch.nn.CrossEntropyLoss()    # mean-squared error for regression
#optimizer = torch.optim.Adam(cov_pred_1.parameters(), lr=learning_rate)
cov_pred_1=cov_pred_1.to(device=device)

optimizer = torch.optim.SGD(cov_pred_1.parameters(), lr=1e-2)#[{"params": lstm1.lstm_output.parameters(), 'lr' : 1e-3,"momentum":0.8},# "betas" : (0.2,0.5)},
                           # {"params": lstm1.fc_input_cell.parameters()},
                           # {"params": lstm1.fc_output_cov.parameters()},{"params": lstm1.lstm.parameters()},*[{"params":lstm1.fc_output_bin[i].parameters()} for i in range(seq_length)]], lr = 1e-2,momentum=0.9)# betas=(0.9,0.99))
#Training
num_data=len(data_set_pre)
medium_loss=0
batchsize=256
for epoch in range(num_epochs+1):
    #classes_labels=random.sample(list(cov_map.keys()),16)
    batch=[]
    for cl in classes_labels:
        data_set_pre_cl=[data_set_pre[i] for i in cov_labels[cl]]
        batch_cl=random.sample(data_set_pre_cl,int(batchsize/16))
        batch.extend(batch_cl)
    #batch=random.sample(data_set_pre,batchsize)
    batch_input=torch.tensor(np.array([np.array(batch[i][0]).astype(np.float32) for i in range(len(batch))]),device=device)
    #batch_cov=torch.stack([torch.nn.functional.one_hot(torch.tensor(np.array(batch[i][1]),device=device),num_classes=num_cov_classes).float() for i in range(batchsize)])
    batch_cov=torch.tensor(np.array([batch[i][1] for i in range(len(batch))]),device=device)
    cov_out = cov_pred_1.forward(batch_input) #forward pass
    loss = criterion(cov_out, batch_cov)
    optimizer.zero_grad() #caluclate the gradient, manually setting to 0
    # obtain the loss function
    loss.backward() #calculates the loss of the loss function       
    medium_loss=medium_loss+loss.item()
    grad_norm=0.
    for p in cov_pred_1.parameters():
        grad_norm=max(grad_norm, p.grad.detach().data.norm(2))
    optimizer.step() #improve from loss, i.e backprop
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f,  grad: %1.5f" % (epoch, medium_loss/100, grad_norm))
        medium_loss=0 

     # scheduler.step()

#Saving the network parameters
torch.save(cov_pred_1.state_dict(),"/home/mpc/LMILP/TrainedModels/cov_pred_model_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_ws.p")
#torch.save(cov_pred_1.state_dict(),'TrainedModels/cov_pred_model_dt01_N20_'+recent_date_str+'.p')

#old format
#torch.save(cov_pred_1.state_dict(), 'cov_pred_model_dt015N15_3.p')



   