#   In the main we just initialize the value of the parameter and then 
# we get the solution from the 
# we call the step function of the environment till the goal is reache
from environment import world
from bincov_predict import bincov_predict
from helper import OA_MIMPC
from bounding import bounding
import numpy as np
import gzip
import time
import os
import re
import datetime
import random
import psutil
import torch
import _pickle as pickle
import _pickle as pickle
import gzip
import pickletools
if __name__=="__main__":
 
    # directory = "/home/mpc/LMILP/Datasets"
    # files = os.listdir(directory)
    # dates = []
    # #N=
    # #dt=
    # for file in files:
    #     #match = re.search("MILP_data_points_dt"+dt+"_N"+N+"_(.*).p", file)
    #     match = re.search("MILP_data_points_dt01_N20_(.*).p", file)
    #     if match:
    #         date_str = match.group(1)
    #         try:
    #             date = datetime.datetime.strptime(date_str,"%Y%m%d-%H%M%S")
    #             dates.append(date)

    #         except:
    #             a=1
    # dates.sort(reverse=True)
    # recent_date_str=dates[0].strftime("%Y%m%d-%H%M%S")
    # recent_date_str=

    N=40
    dt=0.1
    dt_string="01"
    recent_date_str="20230301-144802"
    # Binary solution and cover maps paths
    # PIK="/home/mpc/LMILP/ProcessedFiles/cov_map_clean_pr_dt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_2.p"#N=20
    recent_date_str="20230301-144802"

    data_PIK="/home/mpc/LMILP/ProcessedFiles/data_set_pr_dt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_binaug.p"
    
    #PIK="/home/mpc/LMILP/ProcessedFiles/cov_map_pr_dt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_partial.p"#N=40
    PIK="/home/mpc/LMILP/ProcessedFiles/cov_map_pr_dt01_N40_partial.p"
    #PIK1="ProcessedFiles/bin_map_pr_dt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"#for NN
    #PIK1="/home/mpc/LMILP/ProcessedFiles/bin_map_pr_dt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_updated.p"#For RF N=20
    PIK1="/home/mpc/LMILP/ProcessedFiles/bin_map_pr_dt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_final.p"#For RF N=40

    #Neural Network model Paths
    bin_pred_model_path=[]
    for i in range(N):
        bin_pred_model_path.append("/home/mpc/LMILP/TrainedModels/bin_pred_model_"+str(i)+"_dt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_ws1.p")
    #cov_pred_model_path="/home/mpc/LMILP/TrainedModels/cov_pred_model_dt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_temp.p"
    #Random Forest model paths
    # random_forest_model_path_cov="/home/mpc/LMILP/TrainedModels/rf_modeldt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_cov_2.pkl" N=20
    # random_forest_model_path_bin="/home/mpc/LMILP/TrainedModels/rf_modeldt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_bin.pkl"
    random_forest_model_path_cov="/home/mpc/LMILP/TrainedModels/rf_modeldt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_cov2.pkl" #N=40
    random_forest_model_path_bin="/home/mpc/LMILP/TrainedModels/rf_modeldt"+dt_string+"_N"+str(N)+"_"+recent_date_str+"_bin2.pkl"
    # random_forest_models_path_bin=[]
    # for i in range(N):
    #     random_forest_models_path_bin.append("TrainedModels/rf_model_"+str(i)+"_dt"+dt_string+"_N"+str(N)+"_"+recent_date_str+".pkl")
    
    
    #Maps loading 

    # with gzip.open(PIK, "rb") as f:
    #     p=pickle.Unpickler(f)
    #     cov_map=p.load()
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
      
    P=OA_MIMPC(N=N,dt=dt)
    initial_conditions=[k[0] for k in dataset]
 
    #bincov_predictor=bincov_predict(bin_pred_model_path,cov_pred_model_path,cov_map)
    # hidden_size_cov=254
    # hidden_size_bin=128
    # rep_size_bin=32
    #state_dim=len(dataset[0][0])
    # bin_dim=240
    #bincov_predictor=bincov_predict(bin_pred_model_path,cov_pred_model_path,cov_map,bin_dim=bin_dim)
    #bin_dim=len(bin_map[0][0])*N # for NN
    bin_dim=len(bin_map[0])
    state_dim=4
    hidden_size_cov=64
    depth_cov=4
    # Deep set bin pred model params (Classification)
    hidden_size_phi=64
    hidden_size_rho=64
    rep_size_bin=128
    depth_phi=2
    depth_rho=1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #bincov_predictor=bincov_predict(bin_pred_model_path,cov_pred_model_path,cov_map,bin_dim,depth_phi,depth_rho,depth_cov,state_dim,hidden_size_cov,hidden_size_phi,hidden_size_rho,rep_size_bin,bin_map)
    bounder=bounding(N=N)
    N_test=100
    #bincov_predictor=bincov_predict(bin_pred_model_path,bin_pred_model_path,cov_map,bin_dim,depth_phi,depth_rho,depth_cov,state_dim,hidden_size_cov,hidden_size_phi,hidden_size_rho,rep_size_bin,bin_map,device=device,N_pred=True,N=40)

    env=world()
    x_traj=[[] for i in range(N_test)]
    milp_solve_time_gurobi=[[] for _ in range(N_test)]
    milp_solve_time_mosek=[[] for _ in range(N_test)]
    milp_solve_time_scip=[[] for _ in range(N_test)]
    #milp_solve_time_ecosbb=[[] for _ in range(N_test)]
    milp_solve_time_glpkmi=[[] for _ in range(N_test)]

    u_traj=[[] for i in range(N_test)]
    backups=[]
    ubs_lbs=[]
    our_solve_times=[]
    gurobi_solve_times=[]
    mosek_solve_times=[]
    scip_solve_times=[]
    #ecosbb_solve_times=[]
    glpkmi_solve_times=[]
    gurobi_lim_stats=[[] for _ in range(N_test)]
    mosek_lim_stats=[[] for _ in range(N_test)]
    scip_lim_stats=[[] for _ in range(N_test)]
    efficiency_factor=0.9
    cores_num=psutil.cpu_count()
    speed_up_factor=(1-efficiency_factor)+efficiency_factor/cores_num
    N_prop=4
    for i in range(N_test):
        #x0=P.generate_initial_condition(env.ol,env.ol+env.obs_size)
        x0=random.choice(initial_conditions)+np.random.normal(0,0.001,4)
        #x0=[-2.534957617013888, 1.8238396366129277, 0.6699152366840214, -0.8730660214110519]
        env.set_init(np.array(x0))
        x_traj[i].append(np.array(x0))
        goal=env.check_goal()
        collision=env.check_collision()
        backup=False
        t=0
        while not goal and not collision and t<=N_prop:
            t0=time.time()
            #C_star,y_star_vec=bincov_predictor.predict_sol(env.x.squeeze())
            #y_star=bincov_predictor.predict_binsol_N(env.x.squeeze())
            C_star=cov_map[rf_cov.predict([env.x.squeeze()])[0]]
            y_star=np.array(bin_map[rf_bin.predict([env.x.squeeze()])[0]])
            t1=time.time()
            print("Prediction time:"+str((t1-t0)))
            control,bound, MILP_solvetime_gurobi,MILP_solvetime_mosek,MILP_solvetime_scip,\
            MILP_solvetime_glpkmi,lim_stats_GUROBI,lim_stats_MOSEK,lim_stats_SCIP=bounder.get_sol(C_star,y_star,env.x.squeeze())
            #control,bound, MILP_solvetime=bounder.get_sol(C_star,y_star_vec,env.x.squeeze())
            # if bound[2]=="Backup":
            #     backup=True
            # else:
            
            our_solve_times.append((t1-t0+bound[1]+bound[2])*speed_up_factor)
            gurobi_solve_times.append(MILP_solvetime_gurobi)
            mosek_solve_times.append(MILP_solvetime_mosek)
            scip_solve_times.append(MILP_solvetime_scip)
            #ecosbb_solve_times.append(MILP_solvetime_ecosbb)
            glpkmi_solve_times.append(MILP_solvetime_glpkmi)
            
            print("solve time: "+str((t1-t0+bound[1]+bound[2])*speed_up_factor))
            ubs_lbs.append(bound[0])

            # milp_solve_time_gurobi[i].append(MILP_solvetime_gurobi)
            # milp_solve_time_mosek[i].append(MILP_solvetime_mosek)
            # milp_solve_time_scip[i].append(MILP_solvetime_scip)
            #milp_solve_time_ecosbb[i].append(MILP_solvetime_ecosbb)
            # milp_solve_time_glpkmi[i].append(MILP_solvetime_glpkmi)
            gurobi_lim_stats[i].append(lim_stats_GUROBI)
            mosek_lim_stats[i].append(lim_stats_MOSEK)
            scip_lim_stats[i].append(lim_stats_SCIP)
            print(bound)
            collision, goal = env.step(control)
            x_traj[i].append(env.x)
            u_traj[i].append(control)
            print("iter :"+str(i)+" time :"+str(env.t)+" pos : "+str(env.x.T), " MILP_sol_time_gurobi : "+str(MILP_solvetime_gurobi)+"\n", " MILP_sol_time_mosek : "+str(MILP_solvetime_mosek)," MILP_sol_time_scip : "+str(MILP_solvetime_scip)," MILP_sol_time_glpkmi : "+str(MILP_solvetime_glpkmi))
            
            print("===============================================================")
            t+=1
        if backup:
            backups.append(i)


    gurobi_lim_stats_cat=[]
    [gurobi_lim_stats_cat.extend(k) for k in gurobi_lim_stats]
    mosek_lim_stats_cat=[]
    [mosek_lim_stats_cat.extend(k) for k in mosek_lim_stats]
    scip_lim_stats_cat=[]
    [scip_lim_stats_cat.extend(k) for k in scip_lim_stats]

    sub_gurobi=[k[1] for k in gurobi_lim_stats_cat if k[0] in('optimal','optimal_inaccurate')]
    max_sub_gurobi=np.max(np.array(sub_gurobi))
    avg_sub_gurobi=np.mean(np.array(sub_gurobi))
    min_sub_gurobi=np.min(np.array(sub_gurobi))
    sub_mosek=[k[1] for k in mosek_lim_stats_cat if k[0] in('optimal','optimal_inaccurate')]
    max_sub_mosek=np.max(np.array(sub_mosek))
    avg_sub_mosek=np.mean(np.array(sub_mosek))
    min_sub_mosek=np.min(np.array(sub_mosek))
    sub_scip=[k[1] for k in scip_lim_stats_cat if k[0] in('optimal','optimal_inaccurate')]
    max_sub_scip=np.max(np.array(sub_scip))
    avg_sub_scip=np.mean(np.array(sub_scip))
    min_sub_scip=np.min(np.array(sub_scip))
    infeas_gurobi=1.0-(len(sub_gurobi)/len(gurobi_lim_stats_cat))
    infeas_mosek=1.0-(len(sub_mosek)/len(mosek_lim_stats_cat))
    infeas_scip=1.0-(len(sub_scip)/len(scip_lim_stats_cat))
    infeas_our=len([k for k in our_solve_times if k>0.1])/len(our_solve_times)
    non_backup_solve_times=[our_solve_times,gurobi_solve_times,mosek_solve_times,scip_solve_times,glpkmi_solve_times]     
    #all_solve_times=[milp_solve_time_gurobi,milp_solve_time_mosek,milp_solve_time_scip,milp_solve_time_glpkmi]
    solver_lim_stats=[gurobi_lim_stats,mosek_lim_stats,scip_lim_stats]
    our_suboptimality=ubs_lbs
    trajectories=[x_traj,u_traj]
    PIK="/home/mpc/LMILP/Results/non_backup_solve_timesN40_20230301-144802.p"#+'recent_
    PIK1="/home/mpc/LMILP/Results/all_solve_timesN40_20230301-144802.p"
    PIK2="/home/mpc/LMILP/Results/solver_lim_statsN40_20230301-144802.p"
    PIK3="/home/mpc/LMILP/Results/our_suboptimalityN40_20230301-144802.p"
    PIK4="/home/mpc/LMILP/Results/trajectoriesN40_20230301-144802.p"




    # with gzip.open(PIK, "wb") as f:
    #     pickled=pickle.dumps(k)
    #     optimized_pickle=pickletools.optimize(pickled)
    #     f.write(optimized_pickle)
        

    a=1