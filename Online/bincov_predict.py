# this class will have three methods 
# the first method will use the trained neural network to predict the cover
#after we get the label correponding to a cover we have to decode the label to pickthe corresponding lbs and ubs
# the secon method given a lsit of lbs and ubs will return a prediction of the binary solution
#the third method will be a combination of the two methods and will return the list of predicted lbs and ubs along with the binary solution
import sys
sys.path.append("/home/mpc/LMILP/LAMPOS/")
from Offline.Training.predictions import cov_pred,bin_pred_FC
import torch
import numpy as np

class bincov_predict:
    def __init__(self,bin_pred_model_path,cov_pred_model_path,cov_map,bin_dim,state_dim,hidden_size_cov,cov_depth,bin_map,device=torch.device('cpu'),bin_N=True,N=20) -> None:#bin_map,state_dim,hidden_size_cov,hidden_size_bin,rep_size_bin
        '''
        This class recover the prediciton of both cover and binary solution
        '''
        self.device=device
        self.cov_map=cov_map
        self.bin_map=bin_map
        self.cov_depth=cov_depth
        self.bin_dim=bin_dim
        self.bin_N=bin_N
        self.N=N
        self.cov_classes_dim=len(cov_map.keys())
        self.max_sp_num=max([len(x) for x in self.cov_map.values()])
        self.cov_pred_model=cov_pred(input_size=state_dim,hidden_size=hidden_size_cov,cover_classes_dim=self.cov_classes_dim,depth=self.cov_depth,device=self.device)#18
        self.cov_pred_model.load_state_dict(torch.load(cov_pred_model_path,map_location=self.device))
        if not self.bin_N:
            self.bin_classes_dim=len(bin_map.keys())
            self.bin_pred_model=bin_pred_FC(input_size=state_dim, output_size=self.bin_classes_dim, hidden_size=512,depth=3,device=self.device)
            self.bin_pred_model.load_state_dict(torch.load(bin_pred_model_path,map_location=self.device))

        else:
            self.bin_classes_dim=[len(bin_map[i]) for i in range(N)]
            self.bin_pred_model=[bin_pred_FC(input_size=state_dim, output_size=self.bin_classes_dim[i], hidden_size=64,depth=2,device=self.device) for i in range(N)]
            [self.bin_pred_model[i].load_state_dict(torch.load(bin_pred_model_path[i],map_location=self.device)) for i in range(N)]
            
    def predict_cover(self,x):
        ''' 
        Predict the cover using the trained neural network
        '''
        x=torch.tensor(x.astype(np.float32),device=self.device)
        pred_cover_label=self.cov_pred_model(x)
        pred_cover_label=int(np.argmax(pred_cover_label.cpu().detach()))
        return pred_cover_label
    
    def predict_binsol(self,x,k):#pred_cover
        ''' 
        Predict the binary solution using the FC trained neural network for binary solution
        '''
        input=[torch.tensor((np.array(x).astype(np.float32)))]
        input=torch.stack(input).to(self.device)
        pred_bin=np.array(self.bin_pred_model(input).cpu().detach())
        indices_k = np.argpartition(pred_bin[0], -k)[-k:]
        labels_k = indices_k[np.argsort(pred_bin[0][indices_k])][::-1]
        return labels_k
    
    def predict_binsol_N(self,x):
        ''' 
        Predict the binary solution using the N FC trained neural networks for each time step
        combining the predictions in a single vector representing the solution
        '''
        input=[torch.tensor((np.array(x).astype(np.float32)))]
        input=torch.stack(input).to(self.device)
        pred_bin=[np.argmax(np.array(self.bin_pred_model[i](input).cpu().detach())) for i in range(self.N)]
        binary_solution=[]
        [binary_solution.extend(list(self.bin_map[n][pred_bin[n]])) for n in range(self.N)]#indices_k = np.argpartition(pred_bin[0], -k)[-k:]
        return np.array(binary_solution)
        
    def predict_sol(self,x,bin_N=False):
        pred_cover_label=self.predict_cover(x)
        pred_cover=self.cov_map[pred_cover_label]
        if(self.bin_N):
            pred_bin_sol=self.predict_binsol_N(x)
        else:
            pred_bin_sol=self.predict_binsol(x)
        return pred_cover,pred_bin_sol


