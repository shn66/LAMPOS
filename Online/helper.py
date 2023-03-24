import sys
sys.path.append("/home/mpc/LMILP/LAMPOS/")
import cvxopt.modeling as cpm
import cvxpy as cp
import time
import numpy as np 
import scipy as sp
import copy
import scipy.sparse as spa
import numpy as np
import ray
import psutil



class fast_OA_MIMPC():
    def __init__(self,Q = 1000,
                R = 50, N_obs_max=5, N=20, 
                lb_x=np.array([-3,-3,-2,-2]),
                ub_x=-np.array([-3,-3,-2,-2]),
                lb_u=np.array([-2,-2]),
                ub_u=-np.array([-2,-2]),dt=0.1,MILP=False):
       
        
        self.lb_x=lb_x
        self.lb_u=lb_u
        self.ub_x=ub_x
        self.ub_u=ub_u
        self.N=N
        self.A = np.matrix([[1, 0, dt, 0],[0, 1, 0, dt],[0, 0, 1, 0],[0, 0, 0, 1]])
        self.B = dt*np.matrix([[0, 0],[0, 0],[1, 0],[0, 1]])
        self.nx = self.A.shape[1]#% Number of states
        self.nu = self.B.shape[1]#% Number of inputs
        self.free_x=[]
        self.free_u=np.zeros((self.nu,self.N))
        self.free_cost=0
        self.Q=Q*np.identity(self.nx)
        self.R=R*np.identity(self.nu)
        self.space_dim=2
        self.epsilon_list=[]
        self.epsilon=0
        self.epsilon_max=0.05
        self.ol=np.array([[-2.5,-2.,1, 2 ],[-2.5 ,1,-2, 1.5 ]])
        self.obs_size=np.array([[2.5,2,2, 1],[1.5,2,2.5, 1.5]])
        self.M=2*self.ub_x[0]
        self.N_obs_max=self.ol.shape[1]
        self.ou =self.ol+self.obs_size
        self.MILP=MILP
        # Parameters
        self.x0_cp=cp.Parameter((self.nx),name='x0')
        self.lb_cp=cp.Parameter((self.N_obs_max*2*2*self.N,1),name='lb')
        self.ub_cp=cp.Parameter((self.N_obs_max*2*2*self.N,1),name='ub')
        # Decision variables
        self.x_cp = cp.Variable((self.nx,self.N+1),'x')
        self.u_cp = cp.Variable((self.nu,self.N),'u')
        self.b_l_cp=[cp.Variable((2,self.N),boolean=MILP,name='b_l('+str(i)+')') for i in range(self.N_obs_max)]
        self.b_u_cp=[cp.Variable((2,self.N),boolean=MILP,name='b_u('+str(i)+')') for i in range(self.N_obs_max)]
        self.prob_cp=[]
        self.rng=np.random.default_rng(2)

                 

    def parametric_form_cvxpy(self): 
        
        # MPC problem setup
        
        constraints = [self.x_cp[:,0]==self.x0_cp]
                    
        objective   = 0
        sigma=0
        for k in range(self.N):
            objective = objective + cp.norm(self.Q@self.x_cp[:,k],'inf') + cp.norm(self.R@self.u_cp[:,k],'inf')
            
            constraints += [self.x_cp[:,k+1]== self.A@self.x_cp[:,k] + self.B@self.u_cp[:,k]]           
            constraints+=[self.lb_x<=self.x_cp[:,k+1],
                        self.ub_x>=self.x_cp[:,k+1]]
            constraints+=[self.lb_u<=self.u_cp[:,k],
                        self.ub_u>=self.u_cp[:,k]]

            for i in range (self.N_obs_max):
                constraints+=[
                            self.x_cp[0:2,k+1]<=self.ol[:,i]+self.M*self.b_l_cp[i][:,k],
                            self.x_cp[0:2,k+1]>=self.ou[:,i]-self.M*self.b_u_cp[i][:,k],    
                            self.b_l_cp[i][0,k]+self.b_l_cp[i][1,k]+self.b_u_cp[i][0,k]+self.b_u_cp[i][1,k]<=2*2-1,
                            self.b_l_cp[i][0,k]>=self.lb_cp[sigma],
                            self.b_l_cp[i][0,k]<=self.ub_cp[sigma],
                            ] 
                sigma+=1
        objective = objective + 100*cp.norm(self.Q@self.x_cp[:,k+1],'inf')
        for k in range(self.N):
            for i in range (self.N_obs_max):
                constraints+=[
                self.b_l_cp[i][1,k]>=self.lb_cp[sigma],
                self.b_l_cp[i][1,k]<=self.ub_cp[sigma]]
                sigma+=1
        for k in range(self.N):
            for i in range (self.N_obs_max):
                constraints+=[
                self.b_u_cp[i][0,k]>=self.lb_cp[sigma],
                self.b_u_cp[i][0,k]<=self.ub_cp[sigma]]
                sigma+=1
        for k in range(self.N):
            for i in range (self.N_obs_max):
                constraints+=[
                self.b_u_cp[i][1,k]>=self.lb_cp[sigma],
                self.b_u_cp[i][1,k]<=self.ub_cp[sigma]]
                sigma+=1
        
        self.prob_cp = cp.Problem(cp.Minimize(objective), constraints)


    def assign_parameters(self,x0_p,lbb,ubb):
        self.x0_cp.value=x0_p
        self.lb_cp.value=lbb
        self.ub_cp.value=ubb
        
    
    def solve(self,x0_p,lbb=None,ubb=None,get_bin=False):
        self.x0_cp.value=x0_p
        if(lbb is not None):
            self.lb_cp.value=lbb
            self.ub_cp.value=ubb
            if(self.MILP):
                solver=cp.SCIP
                solution=self.prob_cp.solve(solver=solver,warm_start=True)
            else:
                solver=cp.SCIPY
                solution=self.prob_cp.solve(solver=solver,scipy_options={"method": "highs"},warm_start=True)    
            return_list=[self.x_cp.value, self.u_cp.value, self.prob_cp.value,self.prob_cp.status,self.prob_cp._solve_time]
        else:
            self.lb_cp.value=np.zeros((self.N_obs_max*2*2*self.N,1))
            self.ub_cp.value=np.ones((self.N_obs_max*2*2*self.N,1))
            solver=cp.GUROBI
            solver1=cp.MOSEK
            solver2=cp.SCIP
            solver3=cp.GLPK_MI
            try:
                solution=self.prob_cp.solve(solver=solver,warm_start=False)
                gurobi_time=self.prob_cp.solver_stats.solve_time
                xcp_value1=self.x_cp.value
                ucp_value1=self.u_cp.value
                probcp_value1=solution
                probcp_status1=self.prob_cp.status
            except:
                if self.prob_cp.status=="optimal":
                    solution=self.prob_cp.value
                    gurobi_time=self.prob_cp._solve_time
                    xcp_value1=self.x_cp.value
                    ucp_value1=self.u_cp.value
                    probcp_value1=solution
                    probcp_status1="optimal"
                else:
                    solution=1e15
                    gurobi_time=5.0
                    xcp_value1=None
                    ucp_value1=None
                    probcp_value1=solution
                    probcp_status1="infeasible"
            try:
                solution=self.prob_cp.solve(solver=solver1,warm_start=False)
                mosek_time=self.prob_cp.solver_stats.solve_time
                xcp_value2=self.x_cp.value
                ucp_value2=self.u_cp.value
                probcp_value2=solution
                probcp_status2=self.prob_cp.status
            except:
                solution=1e15
                mosek_time=5.0
                xcp_value2=None
                ucp_value2=None
                probcp_value2=solution
                probcp_status2="infeasible"
            try:
                solution=self.prob_cp.solve(solver=solver2,warm_start=False)
                scip_time=self.prob_cp._solve_time
                xcp_value3=self.x_cp.value
                ucp_value3=self.u_cp.value
                probcp_value3=solution
                probcp_status3=self.prob_cp.status
            except:
                solution=1e15
                scip_time=5.0
                xcp_value3=None
                ucp_value3=None
                probcp_value3=solution
                probcp_status3="infeasible"
            try:
                solution=self.prob_cp.solve(solver=solver3,warm_start=False)
                glpkmi_time=self.prob_cp._solve_time
                xcp_value4=self.x_cp.value
                ucp_value4=self.u_cp.value
                probcp_value4=solution
                probcp_status4=self.prob_cp.status
            except:
                solution=1e15
                glpkmi_time=5.0
                xcp_value4=None
                ucp_value4=None
                probcp_value4=solution
                probcp_status4="infeasible"

            xcp_value_vector=[xcp_value1,xcp_value2,xcp_value3,xcp_value4]
            ucp_value_vector=[ucp_value1,ucp_value2,ucp_value3,ucp_value4]
            probcp_status_vector=[probcp_status1,probcp_status2,probcp_status3,probcp_status4]
            probcp_value_vector=[probcp_value1,probcp_value2,probcp_value3,probcp_value4]
            probcp_value=min(probcp_value_vector)
            probcp_value_index=np.argmin(np.array(probcp_value_vector))
            xcp_value=xcp_value_vector[probcp_value_index]
            ucp_value=ucp_value_vector[probcp_value_index]
            xcp_value=xcp_value_vector[probcp_value_index]
            probcp_status=probcp_status_vector[probcp_value_index]
          
            ## Solutions with time limitations
            try:
                solution=self.prob_cp.solve(solver=solver,TimeLimit=.06,warm_start=False)
                gurobi_time_lim=self.prob_cp.solver_stats.solve_time
                gurobi_status=self.prob_cp.status
                gurobi_lim_cost=self.prob_cp.value
                gurobi_sub=(gurobi_lim_cost-probcp_value)/probcp_value
            except:
                gurobi_time_lim=-1
                gurobi_status='infeasible'
                gurobi_sub=np.nan
            try:
                solution=self.prob_cp.solve(solver=solver1,mosek_params={"MSK_DPAR_MIO_MAX_TIME":.06, \
                    "MSK_IPAR_NUM_THREADS": 4},warm_start=False)
                mosek_time_lim=self.prob_cp.solver_stats.solve_time
                mosek_status=self.prob_cp.status
                mosek_lim_cost=self.prob_cp.value
                mosek_sub=(mosek_lim_cost-probcp_value)/probcp_value
            except:
                mosek_time_lim=-1
                mosek_status='infeasible'
                mosek_sub=np.nan
                

            try:
                solution=self.prob_cp.solve(solver=solver2,scip_params={"limits/time":.06},warm_start=False)
                scip_time_lim=self.prob_cp._solve_time
                scip_status=self.prob_cp.status
                scip_lim_cost=self.prob_cp.value
                scip_sub=(scip_lim_cost-probcp_value)/probcp_value

            except:
                scip_time_lim=-1
                scip_status='infeasible'
                scip_sub=np.nan
            if(scip_sub<-.01 or gurobi_sub<-.01 or mosek_sub<-.01):
                a=1
            return_list=[xcp_value, ucp_value, probcp_value,probcp_status,gurobi_time,mosek_time,scip_time,glpkmi_time,[gurobi_status,gurobi_sub],[mosek_status,mosek_sub],[scip_status,scip_sub]]
        
        if(get_bin):
            bl_values=[]
            bu_values=[]
            for k in range(self.space_dim):
                for i in range(self.N):
                    for j in range(self.N_obs_max):
                        bl_values.append(self.b_l_cp[j][k,i].value)
                        bu_values.append(self.b_u_cp[j][k,i].value)
            return_list.append(bl_values)
            return_list.append(bu_values)
        return return_list




    
    def generate_initial_condition(self,ol_list,ou_list):
        ul_p=self.ub_x[0]
        ll_p=self.lb_x[0]
        x0=self.rng.uniform(ll_p,ul_p,2)
        def check_in_obs(x):
            for i in range(ol_list.shape[1]):
                if (x[0]>=ol_list[0][i] and x[1]>=ol_list[1][i]) and (x[0]<=ou_list[0][i] and x[1]<=ou_list[1][i]):
                    return True
            return False
        while check_in_obs(x0):
            x0=self.rng.uniform(ll_p,ul_p,2)

        return list(x0)+[0.0,0.0]
        
