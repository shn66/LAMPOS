# this class will have three methods 
# the first method will use the trained neural network to predict the cover
#after we get the label correponding to a cover we have to decode the label to pickthe corresponding lbs and ubs
# the secon method given a lsit of lbs and ubs will return a prediction of the binary solution
#the third method will be a combination of the two methods and will return the list of predicted lbs and ubs along with the binary solution
from predictions import cov_pred,bin_pred,bin_pred_FC
import torch
from torch.autograd import Variable
import numpy as np

class bincov_predict:
    def __init__(self,bin_pred_model_path,cov_pred_model_path,cov_map,bin_dim,depth_phi,depth_rho,depth_cov,state_dim,hidden_size_cov,hidden_size_phi,hidden_size_rho,rep_size_bin,bin_map,device=torch.device('cpu'),N_pred=True,N=20) -> None:#bin_map,state_dim,hidden_size_cov,hidden_size_bin,rep_size_bin
        self.device=device
        self.cov_map=cov_map
        self.bin_map=bin_map
        
        self.bin_dim=bin_dim
        self.N=N
        self.cov_classes_dim=len(cov_map.keys())
        self.max_sp_num=max([len(x) for x in self.cov_map.values()])
        #self.cov_pred_model=cov_pred(input_size=state_dim,hidden_size=hidden_size_cov,cover_classes_dim=self.cov_classes_dim,device=self.device)#18
        #self.cov_pred_model=cov_pred(state_dim,hidden_size_cov,self.cov_classes_dim,depth_cov,self.device)#
        #self.cov_pred_model.load_state_dict(torch.load(cov_pred_model_path,map_location=self.device))
        #self.bin_pred_model=bin_pred(hidden_size=hidden_size_bin,rep_size=rep_size_bin,binary_dim=self.bin_dim,state_dim=state_dim,max_sp=self.max_sp_num,device=self.device)
        # self.bin_pred_model=bin_pred(hidden_size_phi,hidden_size_rho,rep_size_bin,bin_dim,self.bin_classes_dim,state_dim,self.max_sp_num,depth_phi,depth_rho,self.device)
        if not N_pred:
            self.bin_classes_dim=len(bin_map.keys())
            self.bin_pred_model=bin_pred_FC(input_size=state_dim, output_size=self.bin_classes_dim, hidden_size=512,depth=3,device=self.device)
            self.bin_pred_model.load_state_dict(torch.load(bin_pred_model_path,map_location=self.device))

        else:
            self.bin_classes_dim=[len(bin_map[i]) for i in range(N)]
            self.bin_pred_model=[bin_pred_FC(input_size=state_dim, output_size=self.bin_classes_dim[i], hidden_size=64,depth=2,device=self.device) for i in range(N)]
            [self.bin_pred_model[i].load_state_dict(torch.load(bin_pred_model_path[i],map_location=self.device)) for i in range(N)]

        #self.bin_pred_model=bin_pred(128,32,cover_classes_dim,4,self.max_sp_num,self.device)

        
            
    # def predict_cover(self,x):
    #     x=torch.tensor(x.astype(np.float32),device=self.device)
    #     pred_cover_label=self.cov_pred_model(x)
    #     pred_cover_label=int(np.argmax(pred_cover_label.cpu().detach()))
        
    #     return pred_cover_label
    def predict_binsol(self,x,k):#pred_cover
        
        #input=torch.tensor(np.array([np.hstack((x, np.array(sp[0]), np.array(sp[1]))) for sp in pred_cover]).astype(np.float32),device=self.device)# the input should be a matrix of dimendion binary_dim*2+state_dim*number of element in the cover
        #input=torch.tensor(np.array([np.hstack((x, np.array(sp[0]), np.array(sp[1]))) for sp in pred_cover]).astype(np.float32),device=self.device)# the input should be a matrix of dimendion binary_dim*2+state_dim*number of element in the cover
        ##input_cover=np.array(list(pred_cover)).reshape((len(pred_cover),2*self.bin_dim))
        ##input_cover=np.vstack((input_cover,np.zeros((self.max_sp_num-len(pred_cover),2*self.bin_dim))))
        #input=[torch.tensor(np.hstack((np.array([x]*self.max_sp_num),input_cover)).astype(np.float32))]
        input=[torch.tensor((np.array(x).astype(np.float32)))]

        #input=[torch.tensor(np.array([np.hstack((x, np.array(list(pred_cover)[j][0]), np.array(list(pred_cover)[j][1]))) if j< len(pred_cover) else np.hstack((x, np.zeros((len(list(pred_cover)[0][1]),1)).squeeze(), np.zeros((len(list(pred_cover)[0][1]),1)).squeeze()))  for j in range(self.max_sp_num)]).astype(np.float32))]
        input=torch.stack(input).to(self.device)
        #pred_bin=torch.sigmoid(self.bin_pred_model(input)).cpu().detach()
        pred_bin=np.array(self.bin_pred_model(input).cpu().detach())
        indices_k = np.argpartition(pred_bin[0], -k)[-k:]
        labels_k = indices_k[np.argsort(pred_bin[0][indices_k])][::-1]
        return labels_k#pred_bin#np.round(np.array(pred_bin)) # ...since we want the first k-highest level scores we return the vector and not just the max
    def predict_binsol_N(self,x):#pred_cover

        input=[torch.tensor((np.array(x).astype(np.float32)))]

        #input=[torch.tensor(np.array([np.hstack((x, np.array(list(pred_cover)[j][0]), np.array(list(pred_cover)[j][1]))) if j< len(pred_cover) else np.hstack((x, np.zeros((len(list(pred_cover)[0][1]),1)).squeeze(), np.zeros((len(list(pred_cover)[0][1]),1)).squeeze()))  for j in range(self.max_sp_num)]).astype(np.float32))]
        input=torch.stack(input).to(self.device)
        #pred_bin=torch.sigmoid(self.bin_pred_model(input)).cpu().detach()
        pred_bin=[np.argmax(np.array(self.bin_pred_model[i](input).cpu().detach())) for i in range(self.N)]
        binary_solution=[]
        [binary_solution.extend(list(self.bin_map[n][pred_bin[n]])) for n in range(self.N)]#indices_k = np.argpartition(pred_bin[0], -k)[-k:]
        # labels_k = indices_k[np.argsort(pred_bin[0][indices_k])][::-1]
        return np.array(binary_solution)#pred_bin#np.round(np.array(pred_bin)) # ...since we want the first k-highest level scores we return the vector and not just the max
        # return pred_bin
    def predict_sol(self,x):
        pred_cover_label=self.predict_cover(x)
        pred_cover=self.cov_map[pred_cover_label]
        pred_bin_labels=self.predict_binsol_N(x)#,k)
        #pred_bin_label_vector=self.predict_binsol(pred_cover,x)
        pred_bin_vector=[]
        [pred_bin_vector.extend(self.bin_map[i][pred_bin_labels[i]]) for i in range(self.N)]
        
        return pred_cover,np.array(pred_bin_vector)#pred_bin_vector


