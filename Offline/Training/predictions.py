import numpy as np
import random
import torch.nn.functional as Fun
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable
# from helper import cov_map,bin_sol_map
device = torch.device(0)
## Feedforward model for cover prediction
class cov_pred(nn.Module):
    def __init__(self, input_size, hidden_size, cover_classes_dim,depth,device):
        super(cov_pred, self).__init__()
        # self.binary_dim = binary_dim #number of classes
        
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.output_size=cover_classes_dim
        self.layers=[]
        self.layers.append(nn.Linear(self.input_size,self.hidden_size,device=device))
        self.layers.append(nn.ReLU())
        for i in range(depth):
            self.layers.append(nn.Linear(self.hidden_size,self.hidden_size,device=device))
            self.layers.append(nn.ReLU())
        #Output Layer
        self.layers.append(nn.Linear(self.hidden_size,self.output_size,device=device))
        self.pred=nn.Sequential(*self.layers)
        self.act=nn.Softmax()
        #self.fc_input_cell = nn.Sequential(nn.Linear(input_size, hidden_size,device=device),nn.ReLU(),nn.Linear(hidden_size, hidden_size,device=device),nn.ReLU())#fully connected 1
        #self.fc_output_cov=nn.Sequential(nn.Linear(hidden_size, cover_classes_dim,device=device))

        
    # in the forward function is declared the structure of the neural network
    def forward(self,x):
        
        out=self.pred(self.act(x))
        #we should consider to add a sigmoid here
        
        return out    

## Deep Set model for binary solution prediction ( regression and classification)
class InvariantModel(nn.Module):
    def __init__(self, phi, rho):
        super().__init__()
        self.phi = phi
        self.rho = rho
        
    def forward(self, x):
        # compute the representation for each data point
        z = self.phi.forward(x)
        # sum up the representations
        rep = torch.sum(z, dim=1, keepdim=False)
        # compute the output
        # if we set the problem as a regression problem we take as output the rounded prediction
        out = self.rho.forward(rep)

        return out

class Phi(nn.Module):
    def __init__(self, input_size, output_size, hidden_size,depth,device):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size=hidden_size
        self.layers=[]
        #Input Layer
        self.layers.append(nn.Linear(self.input_size,self.hidden_size,device=device))
        self.layers.append(nn.ReLU())
        for i in range(depth):
            self.layers.append(nn.Linear(self.hidden_size,self.hidden_size,device=device))
            self.layers.append(nn.ReLU())
        #Output Layer
        self.layers.append(nn.Linear(self.hidden_size,self.output_size,device=device))
        self.pred=nn.Sequential(*self.layers)
        

        #self.pred=nn.Sequential(nn.Linear(self.input_size, hidden_size,device=device), nn.ReLU(),nn.Linear(self.hidden_size, hidden_size,device=device), nn.ReLU(), 
        #                        nn.Linear(hidden_size, self.output_size,device=device))

        
    def forward(self, x):

        out=self.pred(x)
    
        return out
    
class bin_pred_FC(nn.Module):
    def __init__(self, input_size, output_size, hidden_size,depth,device):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size=hidden_size
        self.layers=[]
        #Input Layer
        self.layers.append(nn.Linear(self.input_size,self.hidden_size,device=device))
        self.layers.append(nn.ReLU())
        for i in range(depth):
            self.layers.append(nn.Linear(self.hidden_size,self.hidden_size,device=device))
            self.layers.append(nn.ReLU())
        #Output Layer
        self.layers.append(nn.Linear(self.hidden_size,self.output_size,device=device))
        self.pred=nn.Sequential(*self.layers)

        #self.pred=nn.Sequential(nn.Linear(self.input_size, hidden_size,device=device), nn.ReLU(),nn.Linear(self.hidden_size, hidden_size,device=device), nn.ReLU(), 
        #                        nn.Linear(hidden_size, self.output_size,device=device))

        
    def forward(self, x):

        out=self.pred(x)#torch.round(self.act(self.pred(x)))
    
        return out
class bin_pred(nn.Module):
    def __init__(self,hidden_size_phi,hidden_size_rho,rep_size,binary_dim,binary_classes_dim,state_dim,max_sp,depth_phi,depth_rho,device):
        super(bin_pred, self).__init__()
        
        self.input_size = 2*binary_dim+state_dim #input size ()
        self.rep_size = rep_size
        self.hidden_size_rho = hidden_size_rho
        self.hidden_size_phi= hidden_size_phi#hidden state
        self.binary_classes_dim=binary_classes_dim
        self.max_sp=max_sp
        
        phi = Phi(self.input_size, self.rep_size, self.hidden_size_phi,depth_phi,device=device)
        rho = Phi(self.rep_size, self.binary_classes_dim, self.hidden_size_rho,depth_rho,device=device)
        #rho = Phi(self.rep_size,binary_dim, self.hidden_size_rho,depth_rho,device=device)#regression

        self.net = InvariantModel(phi, rho)
        self.act=nn.Sigmoid()      # MSE loss
    # in the forward function is declared the structure of the neural network
    def forward(self,x):        
        #out=torch.round(self.act(self.net(x))) #MSE Loss
        return self.net(x)

# LSTM model for binary prediction

class lstm_pred(nn.Module):
    def __init__(self, lstm_input_size,input_size, hidden_size, num_layers, seq_length,bin_classes_dim, cover_classes_dim,device):
        super(lstm_pred, self).__init__()
        # self.binary_dim = binary_dim #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm_input_size=lstm_input_size
        self.device=device
        # self.input_bn=nn.BatchNorm2d(4)
        self.fc_input_cell = nn.Sequential(nn.Linear(input_size, lstm_input_size,device=device),nn.ReLU())#fully connected 1
        self.fc_output_bin =  [nn.Sequential(nn.Linear(hidden_size, bin_classes_dim[i],device=device), nn.ReLU()) for i in range(seq_length)]#fully connected last layer
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_output_cov=nn.Sequential(nn.Linear(hidden_size, cover_classes_dim,device=device),nn.ReLU())
        self.lstm_output=nn.LSTM(input_size=2*hidden_size*num_layers,hidden_size=hidden_size,batch_first=True)
    # in the forward function is declared the structure of the neural network
    def forward(self,x):
        outputs=[]
        bin_outs=[]

        # cell_outputs=[]
        hidden_state= Variable(torch.zeros( self.num_layers,x.shape[0], self.hidden_size,device=self.device)) #hidden state
        cell_state= Variable(torch.zeros( self.num_layers, x.shape[0], self.hidden_size,device=self.device)) #internal state
        # hidden_states.append(hidden_state)
        #outputs.append(hidden_state.transpose(0,1))
        # x=x.transpose(1,2)
        # x=x[:,:,:,None]
        # norm_batch=self.input_bn(x)
        # norm_batch=torch.squeeze(norm_batch)
        # norm_batch=norm_batch[:,None,:]
        cell_input=self.fc_input_cell(x)
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

## DECISION TREE model for binary prediction