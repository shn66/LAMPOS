import sys
sys.path.append("/home/mpc/LMILP/LAMPOS/")
from Online.helper import fast_OA_MIMPC
from multiprocessing import Pool
import numpy as np
import psutil

class bounding():
    def __init__(self,N=20,parallel_solve=False) -> None:
        self.P=fast_OA_MIMPC(N=N)
        self.P.parametric_form_cvxpy()
        self.P_MILP=fast_OA_MIMPC(MILP=True,N=N)
        self.P_MILP.parametric_form_cvxpy()
        self.P_MILP_tot=fast_OA_MIMPC(MILP=True,N=N)
        self.P_MILP_tot.parametric_form_cvxpy()
        self.gurobi_solve_times=[]
        self.mosek_solve_times=[]
        self.scip_solve_times=[]
        self.glpkmi_solve_times=[]
        self.gurobi_lim_stats=[]
        self.mosek_lim_stats=[]
        self.scip_lim_stats=[]
        self.LAMPOS_subopt=[]
        self.parallel_solve=parallel_solve

        
    def get_LB(self,sp_list,x0):
        '''
        Given the list of subproblems returns:
        1. A lower bound on the cost function corresponding to the minimum value of the costs among the subproblem 
        2. A list of the feasible subproblems 
        3. The time needed for the lower bound computation
        '''
        gamma_vals=[]
        for x in sp_list:
            gamma_vals.append((np.array(x0),np.array(list(x[0])).reshape((-1,1)),np.array(list(x[1])).reshape((-1,1))))
        
        #Parallel solution 
        if self.parallel_solve:
            cores_num=psutil.cpu_count()
            pool1 = Pool(processes=cores_num)
            sp_values=pool1.starmap(self.P.solve,gamma_vals)
            lb_time=sum([res[4] for res in sp_values])
        else:
            lb_time=0
            sp_values=[]
            for gamma in gamma_vals:
                return_list=self.P.solve(gamma[0],gamma[1],gamma[2])
                lb_time=lb_time+return_list[4]
                sp_values.append(return_list)

        feas_sp_list=[[list(sp_list)[idx],sp[2]] for idx,sp in enumerate(sp_values) if sp[0] is not None]
        feas_sp_list.sort(key=lambda x:x[1])
        LB=feas_sp_list[0][1]
        return LB,feas_sp_list,lb_time
    
    def get_UB(self, list_sp,x0,y_star):
        '''
        Given the list of the feasible subproblems and the predicted binary solution returns:
        1. An upper bound on the cost function 
        2. The  control input correspondent to the lower bound
        3. The time needed for the upper bound computation
        '''
        UB=1e20
        LB=list_sp[0][1]
        control=None
        bound=UB-LB

        _,u_sol,obj,status,sol_time=self.P.solve(x0,y_star.reshape(-1,1),y_star.reshape(-1,1))

        if( status not in ["infeasible","infeasible_inaccurate"]):
            UB=obj
            control=u_sol[:,0].reshape((-1,1))
            ub_time=sol_time
            print('correct_bin')
        if status in ["infeasible","infeasible_inaccurate"] or (UB-LB)/UB>0.2:
            ub_time=0
            for sp_item in list_sp:
                sp=sp_item[0]
                _,u_sol,obj,status,sol_time_sub_MILP=self.P_MILP.solve(x0,np.array(sp[0]).reshape(-1,1),np.array(sp[1]).reshape(-1,1))
                ub_time=ub_time+sol_time_sub_MILP
                if(not status in ["infeasible","infeasible_inaccurate"]):
                    UB=obj
                    control=u_sol[:,0].reshape((-1,1))
                    break
        
        return  UB, control,ub_time

        
    def get_sol(self,sp_list,y_star,x0):
        LB,feas_lp,lb_time=self.get_LB(sp_list=sp_list,x0=x0)
        print("got LB")
        if(LB==None):
            print('wait')
        UB, control,ub_time=self.get_UB(list_sp=feas_lp,x0=x0,y_star=y_star)
        print("got UB : "+str(ub_time))
        bound=[(UB-LB)/UB,ub_time,lb_time,"Ours"]
        solution=self.P_MILP_tot.solve(x0)
        
        self.gurobi_solve_times.append(solution[4])
        self.mosek_solve_times.append(solution[5])
        self.scip_solve_times.append(solution[6])
        self.glpkmi_solve_times.append(solution[7])          
        self.gurobi_lim_stats.append(solution[8])
        self.mosek_lim_stats.append(solution[9])
        self.scip_lim_stats.append(solution[10])

        if (UB-LB)/UB> 0.2:
            inputs=solution[1]
            control=inputs[:,0].reshape((-1,1))
            bound=[(UB-LB)/UB,ub_time,lb_time, "Backup"]
        
        self.LAMPOS_subopt.append(bound[0])

        return control,bound

    def get_solvers_statistics(self):


        max_sub_LAMPOS=np.max(np.array(self.LAMPOS_subopt))
        avg_sub_LAMPOS=np.mean(np.array(self.LAMPOS_subopt))
        min_sub_LAMPOS=np.min(np.array(self.LAMPOS_subopt))
        sub_gurobi=[k[1] for k in self.gurobi_lim_stats if k[0] in('optimal','optimal_inaccurate')]
        max_sub_gurobi=np.max(np.array(sub_gurobi))
        avg_sub_gurobi=np.mean(np.array(sub_gurobi))
        min_sub_gurobi=np.min(np.array(sub_gurobi))
        sub_mosek=[k[1] for k in self.mosek_lim_stats if k[0] in('optimal','optimal_inaccurate')]
        max_sub_mosek=np.max(np.array(sub_mosek))
        avg_sub_mosek=np.mean(np.array(sub_mosek))
        min_sub_mosek=np.min(np.array(sub_mosek))
        sub_scip=[k[1] for k in self.scip_lim_stats if k[0] in('optimal','optimal_inaccurate')]
        max_sub_scip=np.max(np.array(sub_scip))
        avg_sub_scip=np.mean(np.array(sub_scip))
        min_sub_scip=np.min(np.array(sub_scip))
        infeas_gurobi=1.0-(len(sub_gurobi)/len(self.gurobi_lim_stats))
        infeas_mosek=1.0-(len(sub_mosek)/len(self.mosek_lim_stats))
        infeas_scip=1.0-(len(sub_scip)/len(self.scip_lim_stats))
        return max_sub_LAMPOS,avg_sub_LAMPOS,min_sub_LAMPOS,max_sub_gurobi,avg_sub_gurobi,min_sub_gurobi,max_sub_mosek,avg_sub_mosek,\
            min_sub_mosek,max_sub_scip,avg_sub_scip,min_sub_scip,infeas_gurobi,infeas_mosek,infeas_scip