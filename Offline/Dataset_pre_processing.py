import _pickle as pickle
import gzip
import pickletools
from helper import OA_MIMPC
from utils import MILP_data, LP_SP_data
from collections import Counter
import numpy as np
import random
import re
import datetime
import os
from operator import itemgetter
from itertools import groupby

class Dataset_handler:

  def __init__(self,N,dt):
    self.dt=dt
    self.N=N
    directory = "/home/mpc/LMILP/Datasets"
    files = os.listdir(directory)
    integer=int(0.1)
    decimal=int(dt/0.1)
    dt_str=str(integer)+str(decimal)
    dates = []
    for file in files:
        match = re.search("MILP_data_points_dt"+dt_str+"_N"+str(N)+"_(.*).p", file)
        #match = re.search("MILP_data_points_dt_N20_(.*).p", file)
        if match:
            date_str = match.group(1)
            try:
              date = datetime.datetime.strptime(date_str,"%Y%m%d-%H%M%S")
              dates.append(date)

            except:
              a=1
    dates.sort(reverse=True)
    self.recent_date_str=dates[0].strftime("%Y%m%d-%H%M%S")
    self.dataset_path="Datasets/dataset_dt"+dt+"_N"+N+"_"+recent_date_str+".p"
    self.dataset_pr = []
    self.cov_labels={}
    self.bin_labels={}
    self.cov_map={}
    self.bin_map={}
  
  def Labler(self,step_label=False):

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

    
    if not step_label:
      binsol_classes_set=set(binsol_classes_objects)
      binsol_classes_list=list(binsol_classes_set)
      self.bin_map= {binsol_classes_list.index(x):x for x in binsol_classes_set}
      [self.data_set.append([data_ld[i][0],cover_classes_list.index(frozenset(list(data_ld[i][2]))),\
                        binsol_classes_list.index(tuple(data_ld[i][1]))]) for i in range(len(data_ld))]
      self.bin_labels={d[2]:[] for d in self.data_set}
      [self.bin_labels[d[2]].append(i) for i,d in enumerate(self.data_set)]
    else:
      num_binvars=len(binsol_classes_objects[0])
      binsol_N_classes=[[binsol_classes_objects[j][int(num_binvars/N)*i:int(num_binvars/N)*(i+1)] \
                        for j in range(len(binsol_classes_objects))] for i in range(N)]
      binsol_classes_sets=[set(binsol_N_classes[i]) for i in range(N)]
      binsol_classes_list=[list(binsol_classes_sets[i]) for i in range(N)]
      self.bin_map= [{binsol_classes_list[i].index(x):x for x in binsol_classes_sets[i]} for i in range(N)]
      [self.data_set.append([data_ld[j][0],cover_classes_list.index(frozenset(list(data_ld[j][2]))),\
                        [binsol_classes_list[i].index(tuple(data_ld[j][1][int(num_binvars/N)*i:int(num_binvars/N)*(i+1)]))\
                         for i in range(N)]]) for j in range(len(data_ld))]
      self.bin_labels= [{binsol_classes_list[i].index(x):[] for x in binsol_classes_sets[i]} for i in range(N)]
      [[self.bin_labels[n][d[2][n]].append(i) for i,d in enumerate(self.data_set)] for n in range(N)]
    
    self.cov_labels={d[1]:[] for d in self.data_set}
    [self.cov_labels[d[1]].append(i) for i,d in enumerate(self.data_set)]
    
  
  def cov_reassign(self,threshold):
    '''
    This function compress the existing cover classes 
    depending on threshold
    '''
    cov_labels_low=[k for k in self.cov_labels.keys() if len(self.cov_labels[k])<threshold or len(list(self.cov_map[k]))>40]
    cov_labels_high=[k for k in self.cov_labels.keys() if len(self.cov_labels[k])>=threshold and len(list(cov_map[k]))<=40]

    for k in cov_labels_low:

      bin_sols_k=[[bin_map[dataset_pre[i][2]], i] for i in cov_labels[k]]
      max_match_cov_num=0
      max_match_cov_label=None
      for bin_sol_k in bin_sols_k:
        for h in cov_labels_high:
          cov_set_h=cov_map[h]
          for sp in cov_set_h:
            if (np.round(bin_sol_k[0])>=sp[0]).all() and (np.round(bin_sol_k[0])<=sp[1]).all():
                if max_match_cov_num<sum(np.array(sp[0])==np.array(sp[1])):
                    max_match_cov_num=sum(np.array(sp[0])==np.array(sp[1]))
                    max_match_cov_label=h
        try:            
          cov_labels[max_match_cov_label].append(bin_sol_k[1])
        except:
          print("stop")
      
    cov_labels_compressed={k:cov_labels[k] for k in cov_labels_high}

    i=0
    cov_map_compressed={}
    cov_labels_new={}
    for k in cov_labels_compressed:
      cov_map_compressed.update({i:cov_map[k]})
      cov_labels_new.update({i:cov_labels_compressed[k]})
      for d in cov_labels_compressed[k]:
        dataset_pre[d][1]=i
      
      i+=1  
    return []
  def cov_augmenter(self,threshold):
    cov_labels_low_1=[k for k in cov_labels_new.keys() if len(cov_labels_new[k])<=100]
    skip=False
    for k in cov_labels_low_1:
      dataset_pre_k=[dataset_pre[i] for i in cov_labels_new[k]]
      initial_conditions_k=[d[0] for d in dataset_pre_k]
      binary_solutions_k=[[bin_map[d[2]],d[2]] for d in dataset_pre_k] 
      i=0
      std_dev=1e-2
      check=False
      while len(cov_labels_new[k])<50:
        if(i>200 and not check):
          std_dev=std_dev/10
          i=0
          if(std_dev<1e-4):
            dataset_pre.remove(dataset_pre_k[0])
            del cov_labels_new[k]
            skip=True
            break
        problems_number=len(dataset_pre)
        random_x0=random.choice(initial_conditions_k)
        random_ptbd_x0=np.array(random_x0)+np.random.normal(0,std_dev,4)
      
        for b in binary_solutions_k:
          solution=P_MILP.get_LP_sol(random_ptbd_x0,lbb=np.array(b[0]).reshape(-1,1),ubb=np.array(b[0]).reshape(-1,1),get_bin=True)

          if(solution[3]!='infeasible'):
            dataset_pre.append([list(random_ptbd_x0),k,b[1]])
            cov_labels_new[k].append(problems_number)
            bin_labels[b[1]].append(problems_number)
            check=True
            break
        if(skip):
          skip=False
          continue
        i+=1
    return []
  def bin_augmenter(self):
    return []
  def save_processed_dataset(self):
    PIK="/home/mpc/LMILP/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+self.recent_date_str+".p"
    PIK1="/home/mpc/LMILP/ProcessedFiles/cov_map_pr_dt"+dt_str+"_N"+str(N)+"_"+self.recent_date_str+".p"
    PIK2="/home/mpc/LMILP/ProcessedFiles/cov_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+self.recent_date_str+".p"
    PIK3="/home/mpc/LMILP/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+self.recent_date_str+".p"
    PIK4="/home/mpc/LMILP/ProcessedFiles/bin_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+self.recent_date_str+".p"
    return []
  def load_processed_dataset(self):
    return []
    # date = re.search("MILP_data_points_(.*).p", dataset_path)
    # string_date = date.group(1)
    # PIK="data_set_pre_dt015N15_"+string_date+".p"
    # PIK1="cov_map_dt015N15_"+string_date+".p"
    # PIK2="cov_labels_dt015N15_"+string_date+".p"
    # PIK3="bin_labels_dt015N15_"+string_date+".p"
    # PIK4="bin_map_dt015N15_"+string_date+".p"
    # PIK="data_set_dt015N15_bincov_class.p"
    # PIK1="cov_map_dt015N15_bincov_class.p"
    # PIK2="cov_labels_dt015N15_bincov_class.p"
    # PIK3="bin_map_dt015N15_bincov_class.p"
    # PIK4="bin_labels_dt015N15_bincov_class.p"
    #PIK="data_set_pre_dt01N20_trial"+".p"
    # PIK1="cov_map_dt01N20_trial"+".p"
    # PIK2="cov_labels_dt01N20_trial"+".p"
    # PIK3="bin_map_dt01N20_trial"+".p"
    # PIK4="bin_labels_dt01N20_trial"+".p"
    # # # FOR WRITING AFTER THE FIRST DATASET POST PROCESSING
    # with gzip.open(PIK, "wb") as f:
    #   pickled=pickle.dumps(data_set)
    #   optimized_pickle=pickletools.optimize(pickled)
    #   f.write(optimized_pickle)
    # with gzip.open(PIK1, "wb") as f:
    #   pickled=pickle.dumps(cov_map)
    #   optimized_pickle=pickletools.optimize(pickled)
    #   f.write(optimized_pickle)
    # with gzip.open(PIK2, "wb") as f:
    #   pickled=pickle.dumps(cov_labels)
    #   optimized_pickle=pickletools.optimize(pickled)
    #   f.write(optimized_pickle)
    # with gzip.open(PIK3, "wb") as f:
    #   pickled=pickle.dumps(bin_map)
    #   optimized_pickle=pickletools.optimize(pickled)
    #   f.write(optimized_pickle)
    # with gzip.open(PIK4, "wb") as f:
    #   pickled=pickle.dumps(bin_labels)
    #   optimized_pickle=pickletools.optimize(pickled)
    #   f.write(optimized_pickle)


if __name__=="__main__":
  #Post_process=True
  directory = "/home/mpc/LMILP/Datasets"
  dt=0.1
  # integer=int(0.1)
  # decimal=int(dt/0.1)
  # dt_str=str(integer)+str(decimal)
  # files = os.listdir(directory)
  N=40
  # dates = []
  # for file in files:
  #     match = re.search("MILP_data_points_dt"+dt_str+"_N"+str(N)+"_(.*).p", file)
  #     #match = re.search("MILP_data_points_dt_N20_(.*).p", file)
  #     if match:
  #         date_str = match.group(1)
  #         try:
  #           date = datetime.datetime.strptime(date_str,"%Y%m%d-%H%M%S")
  #           dates.append(date)

  #         except:
  #           a=1
  # dates.sort(reverse=True)
  # recent_date_str=dates[0].strftime("%Y%m%d-%H%M%S")
  #if (Post_process):
  #PIK="Datasets/MILP_data_points_dt"+dt+"_N"+N+"_"+recent_date_str+".p"
  #PIK="Datasets/MILP_data_points_dt01_N20_"+recent_date_str+".p"
  # PIK="/home/mpc/LMILP/Datasets/dataset_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
  dataset_path1="/home/mpc/LMILP/Datasets/dataset_dt01_N40_20230301-144802.p"
  dataset_path2="/home/mpc/LMILP/Datasets/dataset_dt01_N40_20230305-112447.p"
  #PIK="Datasets/dataset_dt01_N20_"+recent_date_str+".p"
  #PIK="Datasets/MILP_data_points_dt015N15.p"
  #PIK1="Datasets/new_dataset_dt015N15.p"
  # dataset_pre,cov_map,cov_labels,bin_map,bin_labels=Dataset_pre_processing(dataset_path1,dataset_path2)#
#  dataset_pre,cov_map,cov_labels=Dataset_pre_processing(PIK,PIK1)#bin_map,bin_labels
  # else:
  dt_str="01"
  PIK="/home/mpc/LMILP/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+"partial.p"
  PIK1="/home/mpc/LMILP/ProcessedFiles/cov_map_pr_dt"+dt_str+"_N"+str(N)+"_"+"partial.p"
  PIK2="/home/mpc/LMILP/ProcessedFiles/cov_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+"partial.p"
  PIK3="/home/mpc/LMILP/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+"partial.p"
  PIK4="/home/mpc/LMILP/ProcessedFiles/bin_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+"partial.p"

  # # PIK3="bin_map_dt01N20_trial"+".p"
  # # PIK4="bin_labels_dt01N20_trial"+".p"

  cov_map={}
  with gzip.open(PIK, "rb") as f:
    p=pickle.Unpickler(f)
    dataset_pre=p.load()
  # with gzip.open(PIK1, "rb") as f:
  #   p=pickle.Unpickler(f)
  #   cov_map=p.load()
  with gzip.open(PIK1, "rb") as f1:
      while True:
          try:
              p=pickle.Unpickler(f1)
              cov_list_k=p.load()
              cov_map.update({cov_list_k[0]:cov_list_k[1]})
          except EOFError:
                break
  with gzip.open(PIK2, "rb") as f:
    p=pickle.Unpickler(f)
    cov_labels=p.load()
  with gzip.open(PIK3, "rb") as f:
    p=pickle.Unpickler(f)
    bin_map=p.load()
  with gzip.open(PIK4, "rb") as f:
    p=pickle.Unpickler(f)
    bin_labels=p.load()
 
  
  threshold=1
  P_MILP=OA_MIMPC(MILP=True,N=N,dt=dt,ub_u=np.array([1,1]),lb_u=np.array([-1,-1]))
  P_MILP.parametric_form_cvxpy()
  cov_labels_low=[k for k in cov_labels.keys() if len(cov_labels[k])<threshold or len(list(cov_map[k]))>40]
  cov_labels_high=[k for k in cov_labels.keys() if len(cov_labels[k])>=threshold and len(list(cov_map[k]))<=40]
  # cov_labels_low=[k for k in cov_labels.keys() if  len(list(cov_map[k]))>10]
  # cov_labels_high=[k for k in cov_labels.keys() if len(list(cov_map[k]))<=10]

  for k in cov_labels_low:
    #bin_sols_k=[[d[2],idx] for idx,d in  enumerate(dataset_pre) if d[1]==k]

    bin_sols_k=[[bin_map[dataset_pre[i][2]], i] for i in cov_labels[k]]

    max_match_cov_num=0
    max_match_cov_label=None
    for bin_sol_k in bin_sols_k:
      for h in cov_labels_high:
        cov_set_h=cov_map[h]
        for sp in cov_set_h:
          if (np.round(bin_sol_k[0])>=sp[0]).all() and (np.round(bin_sol_k[0])<=sp[1]).all():
              if max_match_cov_num<sum(np.array(sp[0])==np.array(sp[1])):
                  max_match_cov_num=sum(np.array(sp[0])==np.array(sp[1]))
                  max_match_cov_label=h
      try:            
        cov_labels[max_match_cov_label].append(bin_sol_k[1])
      except:
        print("stop")
    
  cov_labels_compressed={k:cov_labels[k] for k in cov_labels_high}

  i=0
  cov_map_compressed={}
  cov_labels_new={}
  for k in cov_labels_compressed:
    cov_map_compressed.update({i:cov_map[k]})
    cov_labels_new.update({i:cov_labels_compressed[k]})
    for d in cov_labels_compressed[k]:
      dataset_pre[d][1]=i
    
    i+=1  
  a=1   
# Processing 2 : resample for initial condition close to the sampled initial condition
# check if the solution is the same and and augment the datapoints assigning different initial condition but the 
# same cover and the same binary solution for this we can use cvxpy as follows for each x0 in dataset_pre we solve an MILP 
# using the OA using a slightly perturbed x0 we keep  perturbing till we don-t obtain a prefixed number of initial initial condition 
# that correspond to that binary solutionù
cov_labels_low_1=[k for k in cov_labels_new.keys() if len(cov_labels_new[k])<=100]
continua=False
for k in cov_labels_low_1:
  dataset_pre_k=[dataset_pre[i] for i in cov_labels_new[k]]
  initial_conditions_k=[d[0] for d in dataset_pre_k]
  #binary_solutions_k=[d[2] for d in dataset_pre_k]
  binary_solutions_k=[[bin_map[d[2]],d[2]] for d in dataset_pre_k] 
  i=0
  std_dev=1e-2
  check=False
  while len(cov_labels_new[k])<50:
    if(i>200 and not check):
      std_dev=std_dev/10
      i=0
      if(std_dev<1e-4):
         dataset_pre.remove(dataset_pre_k[0])
         del cov_labels_new[k]
         continua=True
         break
    problems_number=len(dataset_pre)
    random_x0=random.choice(initial_conditions_k)
    random_ptbd_x0=np.array(random_x0)+np.random.normal(0,std_dev,4)
  
    for b in binary_solutions_k:
      solution=P_MILP.get_LP_sol(random_ptbd_x0,lbb=np.array(b[0]).reshape(-1,1),ubb=np.array(b[0]).reshape(-1,1),get_bin=True)
      #solution=P_MILP.get_LP_sol(random_ptbd_x0,lbb=np.array(b).reshape(-1,1),ubb=np.array(b).reshape(-1,1),get_bin=True)

      if(solution[3]!='infeasible'):
        # binary_sol=np.concatenate((np.round(solution[4]),np.round(solution[5])))
        # dataset_pre.append([list(random_ptbd_x0),k,binary_sol])

        #dataset_pre.append([list(random_ptbd_x0),k,b])
        dataset_pre.append([list(random_ptbd_x0),k,b[1]])
        cov_labels_new[k].append(problems_number)
        bin_labels[b[1]].append(problems_number)
        check=True
        break
    if(continua):
       continua=False
       continue
        
    i+=1
# Processing 3 : Augmenting the dataset until each binary solution correspond to 
# a minimum number of different datapoints





#for k in cov_labels_low_1:
#  for d in cov_labels_compressed[k]:
#     dataset_pre[d][1]=dataset_pre[cov_labels_compressed[k][0]][1] 

# i=0
# cov_labels_compressed_final={}
# for k in cov_labels_compressed.keys():
#   cov_labels_compressed_final.update({i:cov_labels_compressed[k]})
#   i=i+1
dt_str="01"
recent_date_str="20230301-144802"
#PIK="ProcessedFiles/data_set_pr_dt01_N20_"+recent_date_str+".p"
PIK="/home/mpc/LMILP/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_1.p"
# #PIK1="ProcessedFiles/cov_map_pr_dt01_N20_"+recent_date_str+".p"
PIK1="/home/mpc/LMILP/ProcessedFiles/cov_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_1.p"
# #PIK2="ProcessedFiles/cov_labels_pr_dt01_N20_"+recent_date_str+".p"
PIK2="/home/mpc/LMILP/ProcessedFiles/cov_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_1.p"
# # PIK3="ProcessedFiles/bin_map_pr_dt015N15_bincov_class_tmp.p"
PIK3="/home/mpc/LMILP/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_1.p"
# # PIK4="ProcessedFiles/bin_labels_pr_dt015N15_bincov_class_tmp.p"
PIK4="/home/mpc/LMILP/ProcessedFiles/bin_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_1.p"

# PIK3="bin_map_pr_dt015N15_"+recent_date_str+".p"
# PIK4="bin_labels_pr_dt015N15_"+recent_date_str+".p"
# PIK="ProcessedFiles/data_set_pr_dt015N15_bincov_class_tmp.p"
# PIK1="ProcessedFiles/cov_map_pr_dt015N15_bincov_class_tmp.p"
# PIK2="ProcessedFiles/cov_labels_pr_dt015N15_bincov_class_tmp.p"
# PIK3="ProcessedFiles/bin_map_pr_dt015N15_bincov_class_tmp.p"
# PIK4="ProcessedFiles/bin_labels_pr_dt015N15_bincov_class_tmp.p"
with gzip.open(PIK, "wb") as f:
  pickled=pickle.dumps(dataset_pre)
  optimized_pickle=pickletools.optimize(pickled)
  f.write(optimized_pickle)
with gzip.open(PIK1, "wb") as f:
  pickled=pickle.dumps(cov_map_compressed)
  optimized_pickle=pickletools.optimize(pickled)
  f.write(optimized_pickle)
with gzip.open(PIK2, "wb") as f:
  pickled=pickle.dumps(cov_labels_new)
  optimized_pickle=pickletools.optimize(pickled)
  f.write(optimized_pickle)
a=1
with gzip.open(PIK3, "wb") as f:
  pickled=pickle.dumps(bin_map)
  optimized_pickle=pickletools.optimize(pickled)
  f.write(optimized_pickle)
with gzip.open(PIK4, "wb") as f:
  pickled=pickle.dumps(bin_labels)
  optimized_pickle=pickletools.optimize(pickled)
  f.write(optimized_pickle) 

# num_cov_classes=len(cov_labeled)
# num_bin_classes=[len(binsol_N_labeled[i]) for i in range(N)]

