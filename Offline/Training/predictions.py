import sys
sys.path.append("/home/mpc/LMILP/LAMPOS/")
import numpy as np
import torch.nn.functional as Fun
import torch #pytorch
import torch.nn as nn
device = torch.device(0)



class cov_pred(nn.Module):
    '''
    Feedforward model for cover prediction
    '''
    def __init__(self, input_size, hidden_size, cover_classes_dim,depth,device):
        super(cov_pred, self).__init__()
        self.input_size = input_size    # Input size
        self.hidden_size = hidden_size  # Hidden state
        self.output_size=cover_classes_dim
        self.layers=[]
        self.layers.append(nn.Linear(self.input_size,self.hidden_size,device=device))
        self.layers.append(nn.ReLU())
        for i in range(depth):
            self.layers.append(nn.Linear(self.hidden_size,self.hidden_size,device=device))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_size,self.output_size,device=device))  # Output Layer
        self.pred=nn.Sequential(*self.layers)
        self.act=nn.Softmax()
        
    def forward(self,x):      
        out=self.pred(self.act(x))        
        return out    



class bin_pred_FC(nn.Module):
    '''
    Feed Forwoard Model for binary solution  predictions
    '''
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
    
    def forward(self, x):
        out=self.pred(x)#torch.round(self.act(self.pred(x)))
        return out



class bin_pred(nn.Module):
    '''
    Deep Set model for binary solution prediction ( can be used for both regression and classification)[Currently not used]
    '''
    def __init__(self,hidden_size_phi,hidden_size_rho,rep_size,binary_dim,binary_classes_dim,state_dim,max_sp,depth_phi,depth_rho,device):
        super(bin_pred, self).__init__()
        
        self.input_size = 2*binary_dim+state_dim    # Input size 
        self.rep_size = rep_size
        self.hidden_size_rho = hidden_size_rho
        self.hidden_size_phi= hidden_size_phi       # Hidden state
        self.binary_classes_dim=binary_classes_dim
        self.max_sp=max_sp
        phi = self.Phi(self.input_size, self.rep_size, self.hidden_size_phi,depth_phi,device=device)
        rho = self.Phi(self.rep_size, self.binary_classes_dim, self.hidden_size_rho,depth_rho,device=device)
        #rho = Phi(self.rep_size,binary_dim, self.hidden_size_rho,depth_rho,device=device)#regression
        self.net = self.InvariantModel(phi, rho)

    def forward(self,x):        
        return self.net(x)
    
    class InvariantModel(nn.Module):

        def __init__(self, phi, rho):
            super().__init__()
            self.phi = phi
            self.rho = rho
            
        def forward(self, x):
            z = self.phi.forward(x)                     # Compute the representation for each data point
            rep = torch.sum(z, dim=1, keepdim=False)    # Sum up the representations
            out = self.rho.forward(rep)

            return out

    class Phi(nn.Module):

        def __init__(self, input_size, output_size, hidden_size,depth,device):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.hidden_size=hidden_size
            self.layers=[]
            self.layers.append(nn.Linear(self.input_size,self.hidden_size,device=device))   # Input Layer
            self.layers.append(nn.ReLU())
            for i in range(depth):
                self.layers.append(nn.Linear(self.hidden_size,self.hidden_size,device=device))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(self.hidden_size,self.output_size,device=device))  # Output Layer
            self.pred=nn.Sequential(*self.layers)

        def forward(self, x):
            out=self.pred(x)
            return out

