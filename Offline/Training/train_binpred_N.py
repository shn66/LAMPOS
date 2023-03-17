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
import os
import re
import datetime


directory = "/home/mpc/LMILP/Datasets"
files = os.listdir(directory)
dates = []

# Select the horizon and sampling time

# N=20
# dt=0.1
# integer=int(0.1)
# decimal=int(dt/0.1)
# dt_str=str(integer)+str(decimal)

# #Pick the most recent files with the selected horizon and sampling time

# directory = "/home/mpc/LMILP/Datasets"
# files = os.listdir(directory)
# for file in files:
#     match = re.search("MILP_data_points_dt"+dt_str+"_N"+str(N)+"_(.*).p", file)
#     if match:
#         date_str = match.group(1)
#         try:
#             date = datetime.datetime.strptime(date_str,"%Y%m%d-%H%M%S")
#             dates.append(date)

#         except:
#             a=1
# dates.sort(reverse=True)
# recent_date_str=dates[0].strftime("%Y%m%d-%H%M%S")
dt_str="01"
dt=0.1
recent_date_str="20230301-144802"
N=40
PIK="/home/mpc/LMILP/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"
PIK1="/home/mpc/LMILP/ProcessedFiles/cov_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_final.p"
PIK2="/home/mpc/LMILP/ProcessedFiles/bin_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"
PIK3="/home/mpc/LMILP/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"
# PIK="ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
# PIK1="ProcessedFiles/cov_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
# PIK2="ProcessedFiles/bin_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"# Classification
# PIK2="ProcessedFiles/bin_map_pr_dtpr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"# Classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with gzip.open(PIK, "rb") as f:
    p=pickle.Unpickler(f)
    dataset=p.load()
with gzip.open(PIK1, "rb") as f:
    p=pickle.Unpickler(f)
    cov_map=p.load()

with gzip.open(PIK2, "rb") as f:
    p=pickle.Unpickler(f)
    bin_labels_N=p.load()           #Classification
with gzip.open(PIK3, "rb") as f:
    p=pickle.Unpickler(f)
    bin_map_N=p.load()              #Classification


#Parameters determined by the shape of the data
#del bin_labels_N[482]
#del bin_map_N[482]
#binary_dim=len(dataset[0][2])
bin_pred_model_path=[]
# bad_models_indexes=[24,25,26,27,28,29,34]
for i in range(N):
    bin_pred_model_path.append("/home/mpc/LMILP/TrainedModels/bin_pred_model_"+str(i)+"_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_ws.p")
binary_dim=len(list(bin_map_N[0].values())[0])*N
state_dim=len(dataset[0][0])
binary_classes_dim_N=[len(bin_map_N[n]) for n in range(N)]


# #Training parameters
classes_labels_N=[list(bin_labels_N[n].keys()) for n in range(N)]#Classification
num_epochs=len(dataset)*3
# hidden_size_phi =256#64
# hidden_size_rho=128#64
# rep_size=512#32
# max_sp_num=max([len(x) for x in cov_map.values()])
# depth_phi=2#3
# depth_rho=1
batch_size=256

# Network creation

#bin_pred_1=bin_pred(hidden_size_phi=hidden_size_phi,hidden_size_rho=hidden_size_rho,rep_size=rep_size,\
# binary_classes_dim=binary_classes_dim,state_dim=state_dim,binary_dim=binary_dim,max_sp=max_sp_num,\
# depth_rho=depth_rho,depth_phi=depth_phi,device=device)                                                #Classification
#bin_pred_1=bin_pred(hidden_size_phi,hidden_size_rho, rep_size, binary_dim,binary_classes_dim,state_dim, max_sp_num,depth_phi,depth_rho, device)
load=True
bin_pred_1=[bin_pred_FC(input_size=state_dim, output_size=binary_classes_dim_N[i], hidden_size=64,depth=2,device=device).to(device=device) for i in range(N)]
if  load:
    [bin_pred_1[i].load_state_dict(torch.load(bin_pred_model_path[i],map_location=device)) for i in bad_models_indexes]#range(N)]

# Loss Function and optimizer definition
criterion= torch.nn.CrossEntropyLoss() 
#criterion=nn.BCEWithLogitsLoss()

#criterion = torch.nn.BCELoss(reduction='mean')    # mean-squared error for regression
#bin_pred_1=bin_pred_1.to(device=device)
optimizer = [torch.optim.SGD(bin_pred_1[i].parameters(), lr=1e-2) for i in range(N)]
#optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)
#[{"params": lstm1.lstm_output.parameters(), 'lr' : 1e-3,"momentum":0.8},# "betas" : (0.2,0.5)},
# # {"params": lstm1.fc_input_cell.parameters()},
# # {"params": lstm1.fc_output_cov.parameters()},{"params": lstm1.lstm.parameters()},*[{"params":lstm1.fc_output_bin[i].parameters()} for i in range(seq_length)]], lr = 1e-2,momentum=0.9)# betas=(0.9,0.99))

# Training
num_data=len(dataset)
medium_loss=0
# epoch=0
#torch.autograd.set_detect_anomaly(True)
classes_batches_N=[]
for n in range(N):
    classes_batches=[]
    for cl in classes_labels_N[n]:
        data_set_cl=[dataset[i] for i in bin_labels_N[n][cl]]
        classes_batches.append(data_set_cl)
    classes_batches_N.append(classes_batches)

for  epoch in range(num_epochs+1) :
    #for i in range(batch_size):
        #cycle=int(epoch/len(dataset_set_pre)/batch_size)
        #piece=int((epoch-cycle*len(dataset_set_pre)*batch_size)/(len(dataset_set_pre)*batch_size/10))
        #i_0=int(piece*len(dataset_set_pre)*batch_size/10)
        #i_end=i_0+int(len(dataset_set_pre)*batch_size/10)

    #batch=random.sample(dataset,batch_size)

    ## Ehnanced bach sampling strategy (this operation assume that in each class there are enough datapoints)
    
    # batch_N=[]
    loss=[]
    grad_norm=0.
    for n,m in zip(bad_models_indexes,range(len(bad_models_indexes))):#range(N):
        batch=[]
        for cl in classes_batches_N[n]:
            batch_cl=random.sample(cl,int(batch_size/16))
            batch.extend(batch_cl)
        input=torch.stack([torch.tensor((np.array(batch[i][0]).astype(np.float32))) for i in range(len(batch))]).to(device)
        bin_out=torch.stack([torch.tensor(np.array(batch[i][2][n])) for i in range(len(batch))]).to(device)#if batch[i][2]!=482
        y_pred = bin_pred_1[n].forward(input) #forward pass ).astype(np.float32)
        loss.append(criterion(y_pred,bin_out))
        optimizer[n].zero_grad()
        loss[m].backward()
        medium_loss=medium_loss+(loss[m].item())/len(bad_models_indexes)#N
        
        for p in bin_pred_1[n].parameters():
            grad_norm=max(grad_norm, p.grad.detach().data.norm(2))
        optimizer[n].step()
    if  epoch % 100 == 0 and epoch>0:
        print("Epoch: %d, loss: %1.5f,  grad: %1.5f" % (epoch, medium_loss/100, grad_norm))
        if medium_loss/100<.01 :
            break
        medium_loss=0 

        
    
    # input=[torch.tensor(np.array([np.hstack((np.array(batch[i][0]), np.array(list(cov_map[batch[i][1]])[j][0]), \
    # np.array(list(cov_map[batch[i][1]])[j][1]))) if j< len(cov_map[batch[i][1]]) \
    # else np.hstack((np.array(batch[i][0]), np.zeros((len(list(cov_map[batch[i][1]])[0][1]),1)).squeeze(), np.zeros((len(list(cov_map[batch[i][1]])[0][1]),1)).squeeze())) \
    # for j in range(max_sp_num)]).astype(np.float32)) for i in range(len(batch))]
    #input=[torch.tensor((np.array(batch[i][0]).astype(np.float32))) for i in range(len(batch))]# if batch[i][2]!=482]
    # input=[torch.tensor(np.array([np.hstack((np.array(sample[i][0]), np.array(list(cov_map[sample[i][1]])[j][0]), \
    #         np.array(list(cov_map[sample[i][1]])[j][1]))) \
    #         if j< len(cov_map[sample[i][1]]) \
    #         else \
    #             np.hstack((np.array(sample[i][0]), np.zeros((len(list(cov_map[sample[i][1]])[0][1]),1)).squeeze(), \
    #                 np.zeros((len(list(cov_map[sample[i][1]])[0][1]),1)).squeeze())) \
    #         for j in range(max_sp_num)]).astype(np.float32)) for i in range(batch_size)]
    
    #input=torch.stack(input).to(device)
    #input=input.transpose(0,1)

    #bin_out_label=torch.stack([torch.nn.functional.one_hot(torch.tensor(np.array(batch[i][2]),device=device),num_classes=binary_classes_dim).float() for i in range(batch_size)])
                                                #bin_map[]
    # bin_out=torch.stack([torch.tensor(np.array(classes_labels.index(batch[i][2]))) for i in range(len(batch))]).to(device)#if batch[i][2]!=482
    # y_pred = bin_pred_1.forward(input) #forward pass ).astype(np.float32)
    # loss = criterion(y_pred,bin_out)

    # #caluclate the gradient, manually setting to 0
    # # obtain the loss function
    # optimizer.zero_grad()
    # loss.backward() #calculates the loss of the loss function       
    # medium_loss=medium_loss+loss.item()
    # grad_norm=0.
    # for p in bin_pred_1.parameters():
    #     grad_norm=max(grad_norm, p.grad.detach().data.norm(2))
    # optimizer.step() #improve from loss, i.e backprop

    # #Print Loss and Gradient
    # if  epoch % 100 == 0 and epoch>0:
    #     print("Epoch: %d, loss: %1.5f,  grad: %1.5f" % (epoch, medium_loss/100, grad_norm))
    #     if medium_loss/100<.01 :
    #         break
    #     medium_loss=0 

for i in range(N):
    i_str=str(i)
    torch.save(bin_pred_1[i].state_dict(),bin_pred_model_path[i])
#torch.save(bin_pred_1.state_dict(),'TrainModels/bin_pred_model_dt"+dt+"_N"+N+"_"+recent_date_str+'.p')
# scheduler.step()
     


   