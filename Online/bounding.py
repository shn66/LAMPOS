# This class will have three methods,
# the first method (CVXPY_based) will return a lower bound on the cost function and the feasible subproblems  given the list of sps corresponding to the minimum 
# value of the cost function over all the subproblem 
# the second method (CVXOPT_based) will return an upper bound on the cost and the solution corresponding to the upper bound  given the list of feasible subproblems  applying some rouding strategies
# the third method will be based on the combination of this two methosd in particualar given the upper bound and the lower bound if the differnce 
# is less than a particular threshold then it will return the solution correponding to the lower bound along with a quality measurament being the difference between lower and upper bound
# if the difference between lower and upper bound is more than a particular threshold it will solve again the full MILP problem by scratch using the parameter
# returnig the correspondent solution and still the differencce between lower bound and upper bound as a clue of the lack of quality in the prediction
from helper import OA_MIMPC
from cvxopt import matrix, solvers
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import re
import time
#import psutil
#num_cpus = psutil.cpu_count(logical=False)

class bounding():
    def __init__(self,N=20) -> None:
        self.P=OA_MIMPC(N=N)
        self.P.parametric_form_cvxpy()
        self.P_MILP=OA_MIMPC(MILP=True,N=N)
        self.P_MILP.parametric_form_cvxpy()
        self.P_MILP_tot=OA_MIMPC(MILP=True,N=N)
        self.P_MILP_tot.parametric_form_cvxpy()

        
    def get_LB(self,sp_list,x0):
        gamma_vals=[]
        for x in sp_list:
            gamma_vals.append((np.array(x0),np.array(list(x[0])).reshape((-1,1)),np.array(list(x[1])).reshape((-1,1))))
        #pool1 = Pool(processes=1)
        #pool1.starmap()
        sp_values=[]
        #t0=time.time()
        lb_time=0
        for gamma in gamma_vals:
           return_list=self.P.get_LP_sol(gamma[0],gamma[1],gamma[2])
           lb_time=lb_time+return_list[4]
           sp_values.append(return_list)
        #sp_values,lb_time= self.P.get_LP_sol_parallel(gammalist=gamma_vals)
        #t1=time.time(); lb_time=(t1-t0)
        #lb_time=lb_time/8
        #here lb_time has to be changed
        feas_sp_list=[[list(sp_list)[idx],sp[2]] for idx,sp in enumerate(sp_values) if sp[0] is not None]
        #feas_sp_sol=[sp for sp in sp_values if bool(sp[0])]
        feas_sp_list.sort(key=lambda x:x[1])
        LB=feas_sp_list[0][1]
        return LB,feas_sp_list,lb_time
    
    def get_UB(self, list_sp,x0,y_star):

        UB=1e20
        LB=list_sp[0][1]
        control=None
        bound=UB-LB
        #break_tot=False
        #for y_star in y_star_vec:
        x_sol,u_sol,obj,status,sol_time=self.P.get_LP_sol(x0,y_star.reshape(-1,1),y_star.reshape(-1,1))##->indent

        #u_idxs=np.array([ idx for idx,x in enumerate(var_list) if "u_(" in x.name and ",0)" in x.name])
        if( status not in ["infeasible","infeasible_inaccurate"]):#IS FEASIBLE [MEANING THAT THE SOLUTION THA I OBTAIN FIXING THE INTEGER VARIABLES TO Y STAR IT IS NOT NONE]
          #Somehow I need to comupute the total solution associated to thata integer value and I can assign the upper bound cost and x_int here
            UB=obj
            bound=UB-LB
            control=u_sol[:,0].reshape((-1,1))
            ub_time=sol_time
            print('correct_bin')
            #break
        if status in ["infeasible","infeasible_inaccurate"] or (UB-LB)/UB>0.2:
            ub_time=0
            for sp_item in list_sp:
                sp_sol=sp_item[1]
                sp=sp_item[0]
                x_sol,u_sol,obj,status,sol_time_sub_MILP=self.P_MILP.get_LP_sol(x0,np.array(sp[0]).reshape(-1,1),np.array(sp[1]).reshape(-1,1))
                ub_time=ub_time+sol_time_sub_MILP
                if(not status in ["infeasible","infeasible_inaccurate"]):
                    UB=obj
                    bound=UB-LB
                    control=u_sol[:,0].reshape((-1,1))
                    #break_tot=True
                    break
        
        
        
        #t1=time.time()
        #ub_time=t1-t0#here ub_time should be changed as well
        return  UB, control,ub_time

        
    def get_sol(self,sp_list,y_star,x0):#get_sol(self,sp_list,y_star_vec,x0):
        LB,feas_lp,lb_time=self.get_LB(sp_list=sp_list,x0=x0)
        print("got LB")
        if(LB==None):
            print('wait')
        UB, control,ub_time=self.get_UB(list_sp=feas_lp,x0=x0,y_star=y_star)
        #UB, control,ub_time=self.get_UB(list_sp=feas_lp,x0=x0,y_star=y_star_vec)

        print("got UB : "+str(ub_time))
        bound=[(UB-LB)/UB,ub_time,lb_time,"Ours"]
        eps=0.1
        #t0=time.time()
        solution=self.P_MILP_tot.get_LP_sol(x0)
        #t1=time.time()
        full_solve_time_gurobi=solution[4]#also the full_sol_time has to be changed
        full_solve_time_mosek=solution[5]
        full_solve_time_scip=solution[6]
        #full_solve_time_ecosbb=solution[7]
        full_solve_time_glpkmi=solution[7]

        lim_stats_gurobi=solution[8]#also the full_sol_time has to be changed
        lim_stats_mosek=solution[9]
        lim_stats_scip=solution[10]
        if (UB-LB)/UB> 0.2:
            inputs=solution[1]
            control=inputs[:,0].reshape((-1,1))
            bound=[(UB-LB)/UB,ub_time,lb_time, "Backup"]
        return control,bound, full_solve_time_gurobi,full_solve_time_mosek,full_solve_time_scip,full_solve_time_glpkmi,lim_stats_gurobi,lim_stats_mosek,lim_stats_scip   # here we should return the solution
