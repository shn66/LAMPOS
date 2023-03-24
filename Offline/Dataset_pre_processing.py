import sys
sys.path.append("/home/mpc/LMILP/LAMPOS/")
import _pickle as pickle
import gzip
import pickletools
from Online.helper import fast_OA_MIMPC
from  utils import MILP_data, LP_SP_data,get_recentdate_str
from collections import Counter
import numpy as np
import random



class Dataset_handler:

  def __init__(self,N,dt):
    self.dt=dt
    self.N=N
    dataset_path = "Offline/Datasets"
    self.recent_date_str,self.dt_str=get_recentdate_str(dataset_path=dataset_path,N=N,dt=dt)
    self.P_MILP=fast_OA_MIMPC(N=N,dt=dt)
    self.P_MILP.parametric_form_cvxpy()
    self.dataset_path="Offline/Datasets/dataset_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+".p"
    self.dataset =[]
    self.datset_N=[]
    self.cov_labels={}
    self.bin_labels={}
    self.cov_map={}
    self.bin_map={}
    self.bin_N=False
  
  def assign_labels(self):
    '''
    This function load the most recent dataset for current N and dt and assign label for both cover and binary solutions
    also defining two dictionary indicating the element corresponding to each cover and binary label in the dataset
    and two dictionary linking each label to the correspondent cover and binary solution
    '''
    data_ld=[]
    with gzip.open(self.dataset_path, "rb") as f1:
        while True:
            try:
                p=pickle.Unpickler(f1)
                data_ld.append(p.load())
            except EOFError:
                  break
            
    cover_classes_objects=[frozenset(list(x[2])) for x  in data_ld]
    cover_classes_set=set(cover_classes_objects)
    cover_classes_list=list(cover_classes_set)
    self.cov_map= {cover_classes_list.index(x):x for x in cover_classes_set}
    binsol_classes_objects=[tuple(x[1]) for x  in data_ld]
    
    
    binsol_classes_set=set(binsol_classes_objects)
    binsol_classes_list=list(binsol_classes_set)
    self.bin_map= {binsol_classes_list.index(x):x for x in binsol_classes_set}
    [self.dataset.append([data_ld[i][0],cover_classes_list.index(frozenset(list(data_ld[i][2]))),\
                      binsol_classes_list.index(tuple(data_ld[i][1]))]) for i in range(len(data_ld))]
    self.bin_labels={d[2]:[] for d in self.dataset}
    [self.bin_labels[d[2]].append(i) for i,d in enumerate(self.dataset)] 
    self.cov_labels={d[1]:[] for d in self.dataset}
    [self.cov_labels[d[1]].append(i) for i,d in enumerate(self.dataset)]
    
  
  def cov_reassign(self,threshold,max_cov_length):
    '''
    This function compress the existing cover classes depending on threshold and the maximum 
    desired cover length (shorter covers determine faster solution times)
    '''
    cov_labels_low=[k for k in self.cov_labels.keys() if len(self.cov_labels[k])<threshold or len(list(self.cov_map[k]))>max_cov_length]
    cov_labels_high=[k for k in self.cov_labels.keys() if len(self.cov_labels[k])>=threshold and len(list(self.cov_map[k]))<=max_cov_length]
    for k in cov_labels_low:
      bin_sols_k=[[self.bin_map[self.dataset[i][2]], i] for i in self.cov_labels[k]]
      max_match_cov_num=0
      max_match_cov_label=None
      for bin_sol_k in bin_sols_k:
        for h in cov_labels_high:
          cov_set_h=self.cov_map[h]
          for sp in cov_set_h:
            if (np.round(bin_sol_k[0])>=sp[0]).all() and (np.round(bin_sol_k[0])<=sp[1]).all():
                if max_match_cov_num<sum(np.array(sp[0])==np.array(sp[1])):
                    max_match_cov_num=sum(np.array(sp[0])==np.array(sp[1]))
                    max_match_cov_label=h
        try:            
          self.cov_labels[max_match_cov_label].append(bin_sol_k[1])
        except:
          print("stop")
      
    i=0
    self.cov_labels={k:self.cov_labels[k] for k in cov_labels_high}
    cov_map_compressed={}
    cov_labels_new={}
    for k in self.cov_labels:
      cov_map_compressed.update({i:self.cov_map[k]})
      cov_labels_new.update({i:self.cov_labels[k]})
      for d in self.cov_labels[k]:
        self.dataset[d][1]=i
      i+=1
    self.cov_labels=cov_labels_new
    self.cov_map=cov_map_compressed
  

  def cov_augment(self,threshold):
    '''
    This Function create new datapoints with cover labels to wich
    are assigned less points than a predetermined threshold 
    '''
    cov_labels_low=[k for k in self.cov_labels.keys() if len(self.cov_labels[k])<=threshold]
    skip=False
    for k in cov_labels_low:
      dataset_pre_k=[self.dataset[i] for i in self.cov_labels[k]]
      initial_conditions_k=[d[0] for d in dataset_pre_k]
      binary_solutions_k=[[self.bin_map[d[2]],d[2]] for d in dataset_pre_k] 
      i=0
      std_dev=1e-2
      check=False
      while len(self.cov_labels[k])<threshold:
        if(i>200 and not check):
          std_dev=std_dev/10
          i=0
          if(std_dev<1e-4):
            self.dataset.remove(dataset_pre_k[0])
            del self.cov_labels[k]
            skip=True
            break
        problems_number=len(self.dataset)
        random_x0=random.choice(initial_conditions_k)
        random_ptbd_x0=np.array(random_x0)+np.random.normal(0,std_dev,4)
        for b in binary_solutions_k:
          solution=self.P_MILP.solve(random_ptbd_x0,lbb=np.array(b[0]).reshape(-1,1),\
                                     ubb=np.array(b[0]).reshape(-1,1),get_bin=True)
          if(solution[3]!='infeasible'):
            self.dataset.append([list(random_ptbd_x0),k,b[1]])
            self.cov_labels[k].append(problems_number)
            self.bin_labels[b[1]].append(problems_number)
            check=True
            break
        if(skip):
          skip=False
          continue
        i+=1

  
  def bin_augment(self,low_threshold,high_threshold):
    '''
    This Function first reduce the binary classes eliminating
    classes with less than a lower threshold then create new datapoints for bin labels to wich
    are assigned less points than a predetermined upper threshold 
    '''
    self.bin_labels={k:self.bin_labels[k] for k in self.bin_labels if len(self.bin_labels[k])>=low_threshold}
    for k in self.bin_labels:
      dataset_k=[self.dataset[i] for i in self.bin_labels[k]]
      initial_conditions_k=[d[0] for d in dataset_k]
      cover_labels_k=[d[1] for d in dataset_k]
      binary_solution_k=self.bin_map[k] 
      std_dev=1e-2
      check=False
      i=0
      j=0
      while len(self.bin_labels[k])<high_threshold:
        if(i>200 and not check):
          std_dev=std_dev/10
          check=False
          i=0
          j+=1
          if(j>3):
            solution_check=self.P_MILP.solve(np.array(x0),lbb=np.array(binary_solution_k).reshape(-1,1),ubb=np.array(binary_solution_k).reshape(-1,1),get_bin=True)
            if(solution_check[3]=='infeasible'):
                j=0
                break
            else:
                j=0
        problems_number=len(self.dataset)
        for x0,cover_label in zip(initial_conditions_k,cover_labels_k):
          random_ptbd_x0=np.array(x0)+np.random.normal(0,std_dev,4)
          solution=self.P_MILP.solve(random_ptbd_x0,lbb=np.array(binary_solution_k).reshape(-1,1),ubb=np.array(binary_solution_k).reshape(-1,1),get_bin=True)
          if(solution[3]!='infeasible'):
            self.dataset.append([list(random_ptbd_x0),cover_label,k])
            self.bin_labels[k].append(problems_number)
            self.cov_labels[cover_label].append(problems_number)
            check=True
            break
        i+=1


  def get_bin_N_step(self):
    '''
    This function create the dataset,labels and map  with binary labels divided per time step
    '''
    self.bin_N=True
    dataset_N=[]
    binsol_classes_objects=[self.bin_map[x[2]] for x  in self.dataset if x[2] in self.bin_labels.keys()]
    num_binvars=len(binsol_classes_objects[0])
    binsol_N_classes=[[binsol_classes_objects[j][int(num_binvars/N)*i:int(num_binvars/N)*(i+1)] for j in range(len(binsol_classes_objects))] for i in range(N)]
    binsol_N_sets=[set(binsol_N_classes[i]) for i in range(N)]
    binsol_N_list=[list(binsol_N_sets[i]) for i in range(N)]
    [dataset_N.append([x[0], x[1], [binsol_N_list[n].index(self.bin_map[x[2]][int(num_binvars/N)*n:int(num_binvars/N)*(n+1)])\
                                    for n in range(N)]]) for x in self.dataset if x[2] in  self.bin_labels.keys()]
    bin_labels_N= [{binsol_N_list[i].index(x):[] for x in binsol_N_sets[i]} for i in range(N)]
    [[bin_labels_N[n][d[2][n]].append(i) for i,d in enumerate(dataset_N)] for n in range(N)]
    bin_map_N= [{binsol_N_list[i].index(x):x for x in binsol_N_sets[i]} for i in range(N)]
   
    
    self.dataset=dataset_N
    self.bin_map=bin_map_N
    self.bin_labels=bin_labels_N
  

  def save_processed_dataset(self):
    '''
    This method save the labeled dataset and the dictionaries
    '''
    PIK1="Offline/ProcessedFiles/cov_map_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+".p"
    PIK2="Offline/ProcessedFiles/cov_labels_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+".p"
    if(self.bin_N):
      PIK3="Offline/ProcessedFiles/data_set_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+"_bin_N.p"
      PIK4="Offline/ProcessedFiles/bin_map_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+"_bin_N.p"
      PIK5="Offline/ProcessedFiles/bin_labels_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+"_bin_N.p"
    else:
      PIK3="Offline/ProcessedFiles/data_set_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+".p"
      PIK4="Offline/ProcessedFiles/bin_map_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+".p"
      PIK5="Offline/ProcessedFiles/bin_labels_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+".p"


    with gzip.open(PIK1, "wb") as f:
      for k in self.cov_map.keys():
        pickled=pickle.dumps([k,tuple(self.cov_map[k])])
        optimized_pickle=pickletools.optimize(pickled)
        f.write(optimized_pickle)
    with gzip.open(PIK2, "wb") as f:
      pickled=pickle.dumps(self.cov_labels)
      optimized_pickle=pickletools.optimize(pickled)
      f.write(optimized_pickle)
    with gzip.open(PIK3, "wb") as f:
      pickled=pickle.dumps(self.dataset)
      optimized_pickle=pickletools.optimize(pickled)
      f.write(optimized_pickle)
    with gzip.open(PIK4, "wb") as f:
      pickled=pickle.dumps(self.bin_map)
      optimized_pickle=pickletools.optimize(pickled)
      f.write(optimized_pickle)
    with gzip.open(PIK5, "wb") as f:
      pickled=pickle.dumps(self.bin_labels)
      optimized_pickle=pickletools.optimize(pickled)
      f.write(optimized_pickle)
  

  def load_processed_dataset(self):
      '''
      This method load the dataset and the dictionaries from the location correspondent
      to the most recent date str for the current N and dt
      '''
      PIK="Offline/ProcessedFiles/data_set_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+".p"
      PIK1="Offline/ProcessedFiles/cov_map_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+".p"
      PIK2="Offline/ProcessedFiles/cov_labels_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+".p"
      PIK3="Offline/ProcessedFiles/bin_map_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+".p"
      PIK4="Offline/ProcessedFiles/bin_labels_pr_dt"+self.dt_str+"_N"+str(self.N)+"_"+self.recent_date_str+".p"
      with gzip.open(PIK, "rb") as f:
        p=pickle.Unpickler(f)
        self.dataset=p.load()
      with gzip.open(PIK1, "rb") as f1:
          while True:
              try:
                  p=pickle.Unpickler(f1)
                  cov_list_k=p.load()
                  self.cov_map.update({cov_list_k[0]:cov_list_k[1]})
              except EOFError:
                    break
      with gzip.open(PIK2, "rb") as f:
        p=pickle.Unpickler(f)
        self.cov_labels=p.load()
      with gzip.open(PIK3, "rb") as f:
        p=pickle.Unpickler(f)
        self.bin_map=p.load()
      with gzip.open(PIK4, "rb") as f:
        p=pickle.Unpickler(f)
        self.bin_labels=p.load()


if __name__=="__main__":
  #Post_process=True
  dt=0.1
  N=40
  bin_N=True
  dataset_hndlr=Dataset_handler(N=N,dt=dt)
  dataset_hndlr.assign_labels()
  dataset_hndlr.cov_reassign(threshold=1,max_cov_length=40)
  dataset_hndlr.cov_augment(threshold=50)
  dataset_hndlr.bin_augment(low_threshold=5,high_threshold=64)
  if(bin_N):
     dataset_hndlr.get_bin_N_step()
  dataset_hndlr.save_processed_dataset()
    
     

