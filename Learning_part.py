import _pickle as pickle
import gzip
from Learning_based_MILP import MILP_data, LP_SP_data
PIK="MILP_data_points.p"
PIK1="new_dataset.p"
data=[]
data1=[]
N=12
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch.nn.functional as Fun
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mm = MinMaxScaler()
ss = StandardScaler() 
device = torch.device(0)

# with gzip.open(PIK, "rb") as f:
#     while True:
#         try:
#             p=pickle.Unpickler(f)
#             data.append(p.load())
#         except EOFError:
#             break

# with gzip.open(PIK1, "rb") as f1:
#     while True:
#         try:
#             p=pickle.Unpickler(f1)
#             data1.append(p.load())
#         except EOFError:
#             break
# data_set_N=[]
# MILP_data_points=[]
# MILP_data_points_indexes=[]
# result=[[MILP_data_points.append(x),data.index(x)] for x in data if x not in MILP_data_points]

# cover_classes_objects=[frozenset(list(x.cov_set)) for x  in MILP_data_points]
# binsol_classes_objects=[tuple(x.opt_sp.y) for x  in MILP_data_points]
# num_binvars=len(binsol_classes_objects[0])
# binsol_N_classes=[[binsol_classes_objects[j][int(num_binvars/N)*i:int(num_binvars/N)*(i+1)] for j in range(len(binsol_classes_objects))] for i in range(N)]
# binsol_N_sets=[set(binsol_N_classes[i]) for i in range(N)]
# cover_classes_set=set(cover_classes_objects)
# binsol_classes_set=set(binsol_classes_objects)

# cover_classes_list=list(cover_classes_set)
# binsol_classes_list=list(binsol_classes_set)
# binsol_N_list=[list(binsol_N_sets[i]) for i in range(N)]


# cov_labeled=[[x,cover_classes_list.index(x)] for x in cover_classes_set]
# binsol_labeled=[[x,binsol_classes_list.index(x)] for x in binsol_classes_set]
# binsol_N_labeled=[[[x,binsol_N_list[i].index(x)] for x in binsol_N_sets[i]]for i in range(N)]

# [data_set_N.append([data1[x[1]][0]['x0'],cover_classes_list.index(frozenset(list(data1[x[1]][2]))),[binsol_N_list[i].index(tuple(data1[x[1]][1][int(num_binvars/N)*i:int(num_binvars/N)*(i+1)]))for i in range(N)]]) for x in result]
# data_set_N[0]

PIK="dataset_N.p"
PIK1="cov_labeled.p"
#PIK2="binsol_labeled.p"
PIK2="binsol_N_labeled.p"
with gzip.open(PIK, "rb") as f:
    p=pickle.Unpickler(f)
    dataset_N=p.load()
with gzip.open(PIK1, "rb") as f:
    p=pickle.Unpickler(f)
    cov_labeled=p.load()
# with gzip.open(PIK2, "rb") as f:
#     p=pickle.Unpickler(f)
#     binsol_labeled=p.load()
with gzip.open(PIK2, "rb") as f:
    p=pickle.Unpickler(f)
    binsol_N_labeled=p.load()

#num_cov_classes=len(cover_classes_list)
num_cov_classes=len(cov_labeled)
#num_bin_classes=[len(binsol_N_list[i]) for i in range(N)]
num_bin_classes=[len(binsol_N_labeled[i]) for i in range(N)]

class LSTM1(nn.Module):
    def __init__(self, lstm_input_size,input_size, hidden_size, num_layers, seq_length,bin_classes_dim, cover_classes_dim,device):
        super(LSTM1, self).__init__()
        # self.binary_dim = binary_dim #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm_input_size=lstm_input_size
        self.input_bn=nn.BatchNorm2d(4)
        self.fc_input_cell = nn.Sequential(nn.Linear(input_size, 64,device=device),nn.ReLU(), nn.Linear(64, lstm_input_size,device=device),nn.ReLU())#fully connected 1
        self.fc_output_bin =  [nn.Sequential(nn.Linear(hidden_size, hidden_size,device=device),nn.ReLU(),nn.Linear(hidden_size, bin_classes_dim[i],device=device), nn.ReLU()) for i in range(seq_length)]#fully connected last layer
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_output_cov=nn.Sequential(nn.Linear(hidden_size, cover_classes_dim,device=device),nn.ReLU())
        self.lstm_output=nn.LSTM(input_size=2*hidden_size*num_layers,hidden_size=hidden_size,batch_first=True)
    # in the forward function is declared the structure of the neural network
    def forward(self,x):
        outputs=[]
        bin_outs=[]

        # cell_outputs=[]
        hidden_state= Variable(torch.zeros( self.num_layers,x.shape[0], self.hidden_size,device=device)) #hidden state
        cell_state= Variable(torch.zeros( self.num_layers, x.shape[0], self.hidden_size,device=device)) #internal state
        
        # hidden_states.append(hidden_state)
        #outputs.append(hidden_state.transpose(0,1))
        x=x.transpose(1,2)
        x=x[:,:,:,None]
        norm_batch=self.input_bn(x)
        norm_batch=torch.squeeze(norm_batch)
        norm_batch=norm_batch[:,None,:]
        cell_input=self.fc_input_cell(norm_batch)
        
        for i in range(self.seq_length):
            # outputs.append(hidden_state.transpose(0,1))
            # if i==0:
            output, (hidden_state, cell_state) = self.lstm(cell_input, (hidden_state, cell_state))
            # else:
            #     output, (hidden_state, cell_state) = self.lstm(output, (hidden_state, cell_state))
            # output_fc=self.fc_output_cell[i]
            # cell_output=output_fc(hidden_state)
            # cell_outputs.append(cell_output)
            # outputs.append(output)
            #outputs.append(output)
            
            bin_outs.append(self.fc_output_bin[i](output.reshape((x.shape[0],self.hidden_size))))
               
        # Propagate input through LSTM
        #output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        #hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        # output_sequence=torch.cat(outputs).view(-1,1)
        # out_cov=self.fc_output_cov(output_sequence)
        #cov_input=torch.cat([hidden_state.transpose(0,1),cell_state.transpose(0,1)]).view(x.shape[0],self.num_layers*2,self.hidden_size)
        #cov_input=hidden_state.transpose(0,1)
        #cov_input=torch.cat(outputs,dim=1).view(x.shape[0],self.seq_length*self.num_layers,self.hidden_size)
        cov_input=torch.cat((hidden_state.transpose(0,1),cell_state.transpose(0,1)),dim=1).view(x.shape[0],1,2*self.num_layers*self.hidden_size)

        #cov_input=output
        hidden_state_cov= Variable(torch.zeros( 1,x.shape[0], self.hidden_size,device=device)) #hidden state
        cell_state_cov= Variable(torch.zeros( 1,x.shape[0], self.hidden_size,device=device)) #internal state
        # out_bin=self.fc_output_bin(hidden_state_sequence)
        cov_output,(hidden_state_cov,cell_state_cov)=self.lstm_output(cov_input,(hidden_state_cov,cell_state_cov))
        # out = self.relu(hn)
        # out = self.fc_1(out) #first Dense
        # out = self.relu(out) #relu
        # out = self.fc(out) #Final Output
        return bin_outs,self.fc_output_cov(hidden_state_cov.transpose(0,1).reshape((x.shape[0],self.hidden_size)))
num_epochs = 60000 #1000 epochs

seq_length=12
input_size = 4 #number of features
lstm_input_size=64
hidden_size = 128 #number of features in hidden state
num_layers = 2#number of stacked lstm layers
bin_classes_dim=num_bin_classes
cover_classes_dim=num_cov_classes

lstm1 = LSTM1( lstm_input_size,input_size, hidden_size, num_layers, seq_length, bin_classes_dim, cover_classes_dim,device) #our lstm class 
criterion = torch.nn.CrossEntropyLoss(reduction='mean')    # mean-squared error for regression
#optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)
lstm1=lstm1.to(device=device)
optimizer = torch.optim.Adam([{"params": lstm1.lstm_output.parameters(), 'lr' : 1e-2, "betas" : (0.2,0.5)},
                             {"params": lstm1.fc_input_cell.parameters()},
                             {"params": lstm1.fc_output_cov.parameters()},{"params": lstm1.lstm.parameters()},*[{"params":lstm1.fc_output_bin[i].parameters()} for i in range(seq_length)]], lr = 1e-3, betas=(0.9,0.99))
num_data=len(dataset_N)
# X=[x[0] for x in dataset_N ]
# Y1=[x[1] for x in dataset_N]
# Y2=[x[2] for x in dataset_N]
# X_s=ss.fit_transform(X)
# Y1_s=mm.fit_transform(np.array([Y1]).transpose(1,0))
# Y2_s=mm.fit_transform(Y2)
# X_s=[list(x) for x in X_s]
# Y1_s=[list(x) for x in Y1_s]
# Y2_s=[list(x) for x in Y2_s]
# for i in range(len(dataset_N)):
#     dataset_N[i][0]=X_s[i] 
medium_loss=0
batchsize=4096
for epoch in range(num_epochs+1):
  batch=random.sample(dataset_N,batchsize)
  batch_input=Variable(torch.tensor(np.array([np.array(batch[i][0]).reshape((1,-1)).astype(np.float32) for i in range(batchsize)]),device=device))
  batch_cov=Variable(Fun.one_hot(torch.tensor([batch[i][1] for i in range(batchsize)],device=device),num_classes=cover_classes_dim))
  batch_N_binsol=[Variable(Fun.one_hot(torch.tensor([batch[j][2][i] for j in range(batchsize)],device=device),num_classes=bin_classes_dim[i]))  for i in range(N)]
  binout,covout = lstm1.forward(batch_input) #forward pass
  
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0
  # obtain the loss function
  loss = criterion(covout, batch_cov.float())
  for i in range(N):
    loss=loss+criterion(binout[i],batch_N_binsol[i].float())

 
  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  medium_loss=medium_loss+loss.item()
  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, medium_loss/100))
    medium_loss=0 