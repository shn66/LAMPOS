import sys
sys.path.append("/home/mpc/LMILP/LAMPOS/")
from Online.environment import world
from Online.bincov_predict import bincov_predict
from utils import get_recentdate_str,compute_solution_time
from Online.helper import fast_OA_MIMPC
from Online.bounding import bounding
import numpy as np
import gzip
import time
import psutil
import torch
import _pickle as pickle
import _pickle as pickle
import gzip
if __name__=="__main__":
    
    N=40
    dt=0.1
    
    #Pick the most recent processed files with the selected horizon and sampling time

    dataset_path="Offline/Datasets"
    recent_date_str,dt_str=get_recentdate_str(dataset_path=dataset_path,N=N,dt=dt)

    data_PIK="Offline/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
    PIK="Offline/ProcessedFiles/cov_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
    bin_N=True
    if(bin_N):
        PIK1="Offline/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"
    else:
        PIK1="Offline/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"


    #Neural Network model Paths
    cov_pred_model_path="Offline/TrainedModels/cov_pred_model_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
    if bin_N:
        bin_pred_model_path=[]
        for i in range(N):
            bin_pred_model_path.append("Offline/TrainedModels/bin_pred_model_"+str(i)+"_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p")
    else:
        bin_pred_model_path="Offline/TrainedModels/bin_pred_model_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
    
    #Random Forest model paths
    random_forest_model_path_cov="Offline/TrainedModels/rf_modeldt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_cov.pkl" #N=40
    if(bin_N):
        random_forest_models_path_bin=[]
        for i in range(N):
            random_forest_models_path_bin.append("Offline/Training/TrainedModels/rf_model_"+str(i)+"_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".pkl")
    else:
        random_forest_model_path_bin="Offline/TrainedModels/rf_modeldt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin.pkl"
    
    # Label dictionaries loading
    cov_map={}
    with gzip.open(PIK, "rb") as f1:
      while True:
          try:
              p=pickle.Unpickler(f1)
              cov_list_k=p.load()
              cov_map.update({cov_list_k[0]:cov_list_k[1]})
          except EOFError:
                break
    with gzip.open(data_PIK, "rb") as f:
        p=pickle.Unpickler(f)
        dataset=p.load()
    with gzip.open(PIK1, "rb") as f:
        p=pickle.Unpickler(f)
        bin_map=p.load()

    # Random Forest Models loading
    with open(random_forest_model_path_cov, 'rb') as f:
        rf_cov = pickle.load(f)
    with open(random_forest_model_path_bin, 'rb') as f:
        rf_bin = pickle.load(f)
      
    P=fast_OA_MIMPC(N=N,dt=dt)
    bin_dim=len(bin_map[0])
    state_dim=4
    hidden_size_cov=128
    depth_cov=3


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bincov_predictor=bincov_predict(bin_pred_model_path,cov_pred_model_path,cov_map,bin_dim,state_dim,hidden_size_cov,\
                                    depth_cov,bin_map,device=device,bin_N=bin_N,N=40)
    parallel_solve=False
    bounder=bounding(N=N,parallel_solve=parallel_solve)
    N_test=100

    env=world()
    x_traj=[[] for i in range(N_test)]
    u_traj=[[] for i in range(N_test)]
    backups=[]
    ubs_lbs=[]
    LAMPOS_solve_times=[]

        
    N_prop=4
    for i in range(N_test):
        x0=P.generate_initial_condition(env.ol,env.ol+env.obs_size)
        env.set_init(np.array(x0))
        x_traj[i].append(np.array(x0))
        goal=env.check_goal()
        collision=env.check_collision()
        backup=False
        t=0
        while not goal and not collision and t<=N_prop:
            t0=time.time()
            C_star,y_star_vec=bincov_predictor.predict_sol(env.x.squeeze()) # For NN
            y_star=bincov_predictor.predict_binsol_N(env.x.squeeze())
            # C_star=cov_map[rf_cov.predict([env.x.squeeze()])[0]]            # For RF
            # y_star=np.array(bin_map[rf_bin.predict([env.x.squeeze()])[0]])
            t1=time.time()
            prediction_time=t1-t0
            
            print("Prediction time:"+str(prediction_time))
            control,bound=bounder.get_sol(C_star,y_star,env.x.squeeze())
            lb_time=bound[1]
            ub_time=bound[2]
            solution_time=compute_solution_time(prediction_time,lb_time,ub_time,parallel_solve=parallel_solve)
            LAMPOS_solve_times.append(solution_time)
            print(bound)
            print("solve time: "+str(solution_time))
            collision, goal = env.step(control)
            print("===============================================================")
            x_traj[i].append(env.x)
            u_traj[i].append(control)
            t+=1
        if backup:
            backups.append(i)


    infeas_our=len([k for k in LAMPOS_solve_times if k>2*np.mean(np.array(LAMPOS_solve_times))])/len(LAMPOS_solve_times)
    trajectories=[x_traj,u_traj]
    max_sub_LAMPOS,avg_sub_LAMPOS,min_sub_LAMPOS,max_sub_gurobi,avg_sub_gurobi,min_sub_gurobi,max_sub_mosek,avg_sub_mosek,\
            min_sub_mosek,max_sub_scip,avg_sub_scip,min_sub_scip,infeas_gurobi,infeas_mosek,infeas_scip=bounder.get_solvers_statistics()

