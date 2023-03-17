import cvxopt.modeling as cpm
# from mosek import iparam
from cvxopt import solvers, matrix
import cvxpy as cp
import numpy as np 
import scipy as sp
import scipy.sparse as spa
import numpy as np
import copy
# import ray
import time
import psutil
num_cpus = psutil.cpu_count(logical=False)
#ray.init(num_cpus=num_cpus)


class LP_SP_Builder:
    def __init__(self):
        self.constraintList = []
        self.str2constr = {}
        self.constraint_dict={}
        self.objective=[]
        self.prob=[]
    def solve__(self):
        self.prob=cpm.op(self.objective,self.constraintList)

        self.prob.solve()
        def inMatrixForm():
            lp1, vmap, mmap =self.prob._inmatrixform()
            variables = lp1.variables()
            if not variables: 
                raise TypeError('lp must have at least one variable')
            x = variables[0]
            variableList=[v for v in list(vmap.keys())]
            constr_list=[c.name for c in list(mmap.keys())]
            c = lp1.objective._linear._coeff[x]
            d=lp1.objective._constant
            inequalities=lp1._inequalities
            if not inequalities:
                raise TypeError('lp must have at least one inequality')
            G = inequalities[0]._f._linear._coeff[x]
            h = -inequalities[0]._f._constant

            equalities = lp1._equalities
            if equalities:
                A = equalities[0]._f._linear._coeff[x]
                b = -equalities[0]._f._constant
            return x,c,d,G,h,A,b,variableList,constr_list, vmap
        x,c,d,G,h,A,b,variableList,constr_list, vmap=inMatrixForm()
        c_m=matrix(np.array(c)).T; G_m=matrix(np.array(G)); h_m=matrix(np.array(h))
        A_m=matrix(np.array(A)); b_m=matrix(np.array(b))
        solvers.options["show_progress"]=False
        sol=solvers.lp(c_m,G_m,h_m,A_m,b_m)      
        x_sol=sol["x"]

        return x_sol,c,d,G,h,A,b,variableList,constr_list, vmap
      
             
    def reset(self):
        self.constraintList = []
        self.variableList=[]
        self.str2constr = {}
        self.constraint_dict={}
        self.objective=[]
        self.prob=[]
    
    
    def addVar(self,vtype,name,lb,ub):
        x=cpm.variable(name=name)
        self.variableList.append(x)
        return x
    
    

    def setObjective(self,cost,criteria):

        if criteria=='minimize':
            self.objective=cost
        elif criteria=='maximize':
            self.objective==-cost

    def addCons(self, expr, name):
        self.constraintList.append(expr)
        #expr.name=name
        self.str2constr[name] = len(self.constraintList)-1
        self.constraint_dict[name]=expr

    def getConstrList(self):
        return self.constraintList
    
    def getConstrDict(self):
        return self.constraint_dict

    def getConstr(self, str_):
        return self.constraintList[self.str2constr[str_]]


class base_prob:

    def __init__(self):
        self.model=[]#Model()
        self.vars=vars()
        self.opt_vars=[]
        self.opt_vars_sp=[]
        self.LP_SP=LP_SP_Builder()
        self.N=1
        self.N_obs_max=1
        self.space_dim=1
        
            
    def solve_(self):
        
        return{}
    def get_binary_idx(self,i,j,k, bu):
        return i*(self.N*self.N_obs_max)+j*self.N_obs_max+k +bu*(self.space_dim*self.N*self.N_obs_max)
   
    def addVariable(self,dimensions,var_type,var_string,lb=None, ub=None):
        # dimensions= [size of variable per stage, number of stages, number of variables per stage]
        n_dim=len(dimensions)
        self.vars[var_string]={}
        x=self.vars[var_string]
        m=self.LP_SP

              
        if n_dim==1:
            for i in range(dimensions[0]):
                x[i] = m.addVar( vtype=var_type,name=var_string+"(%s)"%(i),lb=None,ub=None)
        elif n_dim==2:
            for i in range(dimensions[0]):
                for j in range(dimensions[1]):                
                    x[i,j] = m.addVar( vtype=var_type,name=var_string+"(%s,%s)"%(i,j),lb=None,ub=None)
        elif n_dim==3:
            for i in range(dimensions[0]):
                for j in range(dimensions[1]):
                    for k in range(dimensions[2]):
                        var_name=var_string+"(%s,%s,%s)"%(i,j,k)
                        
                        if "bu" in var_string:
                            bu=1
                        else:
                            bu=0
                        lb_bnd=lb[self.get_binary_idx(i,j,k,bu)]
                        ub_bnd=ub[self.get_binary_idx(i,j,k,bu)]
                        x[i,j,k] = m.addVar( vtype=var_type,name=var_string+"(%s,%s,%s)"%(i,j,k),lb=None,ub=None)
                        if (var_name=="bl(%s,%s,%s)"%(i,j,k) or var_name=="bu(%s,%s,%s)"%(i,j,k)): 
                                    # Cases :  b_opt=={0,1},                0<b_opt<1
                                    #     lb_bnd=ub_bnd, lb_bnd<ub_bnd      lb_bnd<ub_bnd
                            
                            m.addCons(float(lb_bnd)<=x[i,j,k],name="c_lb"+var_name)
                            m.addCons(float(ub_bnd)>=x[i,j,k],name="c_ub"+var_name)
                            
                
       
        else:
            raise Exception("More than 3 dimensions not supported currently")

        self.opt_vars_sp.append(self.vars[var_string])

class OA_MIMPC(base_prob):
    def __init__(self,Q = 1000,
                R = 50, N_obs_max=5, N=20, 
                lb_x=np.array([-3,-3,-2,-2]),
                ub_x=-np.array([-3,-3,-2,-2]),
                lb_u=np.array([-2,-2]),
                ub_u=-np.array([-2,-2]),dt=0.1,MILP=False):

        super(OA_MIMPC,self).__init__()
        
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
        # self.ol=np.array([[-2.8,-2.5,-1,1.2],[-2.8 ,0.5,-3, -2.5]])
        # self.obs_size=np.array([[1.6,3.5,2,1.5],[3,2.1,2, 5]])
        self.ol=np.array([[-2.5,-2.,1, 2 ],[-2.5 ,1,-2, 1.5 ]])
        self.obs_size=np.array([[2.5,2,2, 1],[1.5,2,2.5, 1.5]])
        self.M=2*self.ub_x[0]
        self.N_obs_max=self.ol.shape[1]
        self.ou =self.ol+self.obs_size
        self.MILP=MILP
              
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

                 

#     def add_variables(self, lb=None, ub=None):
#         self.free_x=[]
#         self.free_cost=0
  
#         self.opt_vars_sp=[]
#         self.LP_SP.reset()
#         self.opt_vars=[]
#         # self.addVariable([self.N], 'C','tx')
#         # self.addVariable([self.N], 'C','tu')

#         self.addVariable([self.nx, self.N+1], 'C','x_')
#         self.addVariable([self.nu, self.N], 'C','u_')
        
#         self.addVariable([self.space_dim, self.N, self.N_obs_max],lb=lb, ub=ub, var_type='B',var_string='bl')
#         self.addVariable([self.space_dim, self.N, self.N_obs_max],lb=lb, ub=ub, var_type='B',var_string='bu')

        
        
                  
#     def add_cost(self):
#         Q=np.squeeze(np.asarray(self.Q))
#         for i in range(self.N):
#             self.free_cost+=np.linalg.norm((1+100*(i==self.N-1))*np.dot(Q,self.free_x[i,:]),np.inf)
     
#         cost=0

#         m=self.LP_SP
#         # tx,tu,x,u,bl,bu=self.opt_vars_sp
#         x,u,bl,bu=self.opt_vars_sp
#         for k in range(self.N):
#             # cost+=tx[k]+tu[k]
#             sum_x=[abs(cpm.sum([(1+100*(k==self.N-1))*float(self.Q[i,j])*x[j,k+1] for j in range(self.nx)])) for i in range(self.nx)]
#             sum_u=[abs(cpm.sum([float(self.R[i,j])*u[j,k] for j in range(self.nu)])) for i in range(self.nu)]
#             cost+=cpm.max(sum_x)+cpm.max(sum_u)


                    
#         m.setObjective(cost,"minimize")
        
#     def add_system_constraints(self, x0):
#         self.x0=x0
#         x_k=x0
#         A=np.squeeze(np.asarray(self.A))
#         self.free_x.append(x_k)
#         for k in range(self.N):
#             x_k=np.dot(A,x_k)
#             self.free_x.append(x_k)    
#         self.free_x=np.asarray(self.free_x)

        
#         m=self.LP_SP
#         # tx,tu,x,u,bl,bu=self.opt_vars_sp
#         x,u,bl,bu=self.opt_vars_sp

        
#         for k in range(self.N+1):
#             for i in range(self.nx):
#                 if k==0:
#                     m.addCons(x[i,k]<=float(self.ub_x[i]), name="c_x_ub_(%s,%s)"%(i,k))
#                     m.addCons(x[i,k]>=float(self.lb_x[i]), name="c_x_lb_(%s,%s)"%(i,k))
#                 if k>0:

# #                     m.addCons((1+100*(k==self.N))*float(self.Q[i,i])*x[i,k]-tx[k-1]<=0, name="c_cost_ub_(%s)_tx_1(%s)"%(i,k-1))
# # #                     m.addCons((1+100*(k==self.N))*float(self.Q[i,i])*x[i,k]-tx[k-1]>=(1+100*(k==self.N))*float(self.Q[i,i])*float(self.lb_x[i])-(1+100*(k==self.N))*float(self.Q[i,i])*float(ub_maxx), name="c_cost_lb_(%s)_tx_1(%s)"%(i,k-1))
# # #                     m.addCons((1+100*(k==self.N))*float(self.Q[i,i])*x[i,k]+tx[k-1]<=(1+100*(k==self.N))*float(self.Q[i,i])*float(self.ub_x[i])+(1+100*(k==self.N))*float(self.Q[i,i])*float(ub_maxx), name="c_cost_ub_(%s)_tx_2(%s)"%(i,k-1))       
# #                     m.addCons((1+100*(k==self.N))*float(self.Q[i,i])*x[i,k]+tx[k-1]>=0, name="c_cost_lb_(%s)_tx_2(%s)"%(i,k-1))
# #                     if i<self.nu:
# #                         m.addCons(float(self.R[i,i])*u[i,k-1]-tu[k-1]<=0, name="c_cost_ub_(%s)_tu_1(%s)"%(i,k-1))
# # #                         m.addCons(float(self.R[i,i])*u[i,k-1]-tu[k-1]>=float(self.R[i,i])*float(self.lb_u[i])-float(self.R[i,i])*float(ub_maxx), name="c_cost_lb_(%s)_tu_1(%s)"%(i,k-1))    
# # #                         m.addCons(float(self.R[i,i])*u[i,k-1]+tu[k-1]<=float(self.R[i,i])*float(self.ub_u[i])+float(self.R[i,i])*float(ub_maxu), name="c_cost_ub_(%s)_tu_2(%s)"%(i,k-1))
# #                         m.addCons(float(self.R[i,i])*u[i,k-1]+tu[k-1]>=0, name="c_cost_lb_(%s)_tu_2(%s)"%(i,k-1))                         
                          
#                     m.addCons(x[i,k]<=float(self.ub_x[i]), name="c_x_ub_(%s,%s)"%(i,k))
#                     m.addCons(x[i,k]>=float(self.lb_x[i]), name="c_x_lb_(%s,%s)"%(i,k))
#                     if i<self.nu:#assuming nu<nx
#                         m.addCons(u[i,k-1]<=float(self.ub_u[i]), name="c_u_ub_(%s,%s)"%(i,k-1))
#                         m.addCons(u[i,k-1]>=float(self.lb_u[i]), name="c_u_lb_(%s,%s)"%(i,k-1))  

#         for k in range(self.N+1):
#             for i in range(self.nx):
#                 if k==0:
#                     m.addCons(x[i,k]==float(x0[i]), name="c_init_x0_eq(%s)"%i)
#                 else:                 
#                     m.addCons(x[i,k]==cpm.sum([float(self.A[i,j])*x[j,k-1] for j in range(self.nx)])\
#                                         +cpm.sum([float(self.B[i,j])*u[j,k-1] for j in range(self.nu)]), name="c_dyn_eq(%s,%s)"%(i,k-1))

#     def add_OA_constraints(self):

#         M=self.M     
#         self.N_obs_max=self.ol.shape[1]
        
         
#         m=self.LP_SP
#         # tx,tu,x,u,bl,bu=self.opt_vars_sp
#         x,u,bl,bu=self.opt_vars_sp
# #         lb_minx=min(self.lb_x)
# #         ub_maxx=max(self.ub_x)
# #         lb_minu=min(self.lb_u)
# #         ub_maxu=min(self.ub_u)
#         for k in range(self.N):
#             for i in range(self.space_dim):
#                 for l in range(self.N_obs_max):
#                     m.addCons(x[i,k+1]-float(M)*bl[i,k,l]<=float(self.ol[i,l]), name="c_oa_ub_(%s,%s,%s)"%(i,k,l))
# #                     m.addCons(x[i,k+1]-float(M)*bl[i,k,l]>=float(self.lb_x[i])-float(M), name="c_oa_lb_1(%s,%s,%s)"%(i,k,l))  
                    
# #                     m.addCons(x[i,k+1]+float(M)*bu[i,k,l]<=float(self.ub_x[i])+float(M), name="c_oa_ub_2(%s,%s,%s)"%(i,k,l))
#                     m.addCons(x[i,k+1]+float(M)*bu[i,k,l]>=float(self.ou[i,l]), name="c_oa_lb_(%s,%s,%s)"%(i,k,l))
                    
                    
#                     if i==0:
#                         m.addCons(bl[0,k,l]+bl[1,k,l]+bu[0,k,l]+bu[1,k,l]<=2*2-1, name="c_oa_(%s,%s)"%(k,l))
# #                         m.addCons(bl[0,k,l]+bl[1,k,l]+bu[0,k,l]+bu[1,k,l]>=0, name="c_oa_lb_c(%s,%s)"%(k,l))
                        
        
     



#     def retreive_sol_cvxopt(self):

    
#         m=self.LP_SP
#         # tx,tu,x,u,bl,bu=self.opt_vars_sp
#         x,u,bl,bu=self.opt_vars_sp
#         return_dict={}
#         binary_sol={}
#         LPsol_full={}
#         Variables=m.prob.variables()
#         lambda_vals=[float(m.constraint_dict[c].multiplier.value[0]) for c in m.constraint_dict.keys() if "=" not in c]
#         # return_dict.update({"lambda_sol":np.array([float(c.multiplier.value[0]) for c in m.constraintList])})
#         return_dict.update({"lambda_sol":np.array(lambda_vals)})
#         objval=m.prob.objective.value()    
                
                

#         if(objval==None): 
#             status="infeasible"
          
#             return_dict.update({"status":status})      
#             return_dict.update({"optimal_cost":objval})
#             return_dict.update({"x_sol":self.free_x.reshape((-1,1))})
#             return_dict.update({"u_sol":self.free_u.reshape((-1,1))})
#             return_dict.update({"binary_sol":np.zeros(2*(len(bl),1))})
#             return_dict.update({"optimal_cost":self.free_cost})
            
#         else:
#             status="optimal"
#             return_dict.update({"status":status})
#             #objval = m.getPrimalbound()
#             objval=float(m.prob.objective.value()[0])
#             return_dict.update({"optimal_cost":objval})
#             return_dict.update({"x_sol":np.array([float(x_i.value[0]) for x_i in x.values()])})
#             return_dict.update({"u_sol":np.array([float(u_i.value[0]) for u_i in u.values()])})
#             return_dict.update({"b_l_sol":np.array([float(bl_i.value[0]) for bl_i in bl.values()])})
#             return_dict.update({"b_u_sol":np.array([float(bu_i.value[0]) for bu_i in bu.values()])})
            
#             return_dict.update({"x_sol_dict":{xi.name:float(xi.value[0]) for xi in x.values()}})
#             return_dict.update({"u_sol_dict":{ui.name:float(ui.value[0]) for ui in u.values()}})

             
                        
#             for i in range(self.space_dim):
#                 for j in range(self.N):
#                     for k in range(self.N_obs_max):
#                         binary_sol[bl[i,j,k].name]=float(bl[i,j,k].value[0])

#             for i in range(self.space_dim):
#                 for j in range(self.N):
#                     for k in range(self.N_obs_max):
#                         binary_sol[bu[i,j,k].name]=float(bu[i,j,k].value[0])
          
#             return_dict.update({"binary_sol":binary_sol})
#             for var in Variables:
#                 LPsol_full[var.name]=float(var.value[0])
            
#             return_dict.update({"LPsol_full":LPsol_full})
        



#         return return_dict

#     def solve_(self, x0=np.array([0, -2,0,0]),
#                     lb=None, ub=None):
        
            
        

#         self.add_variables(lb=lb, ub=ub)
#         self.add_OA_constraints()
#         self.add_system_constraints(x0)
#         self.add_cost()
#         x,c,d,G,h,A,b,var_list,constr_list, vmap=self.LP_SP.solve__()
#         P_sol=self.retreive_sol_cvxopt()
 
#         return P_sol,x,c,d,G,h,A,b,var_list,constr_list, vmap
    
    def parametric_form_cvxpy(self): 
        
        # MPC problem setup
        Model=[]
        
        
        constraints = [self.x_cp[:,0]==self.x0_cp]
                    
        objective   = 0
        sigma=0
        for k in range(self.N):
            objective = objective + cp.norm(self.Q@self.x_cp[:,k],'inf') + cp.norm(self.R@self.u_cp[:,k],'inf')
            
            constraints += [self.x_cp[:,k+1]== self.A@self.x_cp[:,k] + self.B@self.u_cp[:,k]]
            
            state_constr=[]
            input_constr=[]
            
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
        
        # objective = objective + 100*cp.norm(self.Q@self.x_cp[:,k+1],'inf')
        self.prob_cp = cp.Problem(cp.Minimize(objective), constraints)

        #Variables=[x,u,b_l, b_u]
        #Parameters=[x0,lb,ub]

    def assign_parameters(self,x0_p,lbb,ubb):
        self.x0_cp.value=x0_p
        self.lb_cp.value=lbb
        self.ub_cp.value=ubb
    
    # @ray.remote
    # def get_LP_sol_sin(prob):
    #     solution=prob.solve()

        # return prob.var_dict['x'].value,prob.var_dict['u'].value,prob.value,prob.status
   
    def get_LP_sol(self,x0_p,lbb=None,ubb=None,get_bin=False):
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
            #start=time.time()
            
            #end=time.time()
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
            # ecos_bb_time=self.prob_cp._solve_time
            # solution=self.prob_cp.solve(solver=solver4)
            
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
                solution=self.prob_cp.solve(solver=solver1,mosek_params={"MSK_DPAR_MIO_MAX_TIME":.06, "MSK_IPAR_NUM_THREADS": 4},warm_start=False)
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
        # if (self.MILP):
        #     if(ubb is not None):
        #         solver=cp.GLPK_MI
        #     else:
        #         solver=cp.GUROBI
        # else:
        #     solver=cp.OSQP
        
        # solution=self.prob_cp.solve(warm_start=True,solver=solver)
        # return_list=[self.x_cp.value, self.u_cp.value, self.prob_cp.value,self.prob_cp.status,self.prob_cp.solver_stats.solve_time]
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
    # def parallel_get_LP_sol(self,gammalist):
    #     dasklist = [dask.delayed(self.get_LP_sol)(gamma[0],gamma[1],gamma[2]) for gamma in gammalist]
    #     results = dask.compute(*dasklist, scheduler='processes')
    #     return results
       # def parallel_get_LP_sol(self,gammalist):
    #     dasklist = [dask.delayed(self.get_LP_sol)(gamma[0],gamma[1],gamma[2]) for gamma in gammalist]
    #     results = dask.compute(*dasklist, scheduler='processes')
    #     return results
    # def get_LP_sol_parallel(self,gammalist):
    #     res_id=[]
    #     for gamma in gammalist:
    #         self.assign_parameters(gamma[0],gamma[1],gamma[2])
    #         prob=copy.copy(self.prob_cp)
    #         res_id.append(OA_MIMPC.get_LP_sol_sin.remote(prob))
    #     start=time.time()
    #     results=ray.get(res_id)
    #     end=time.time()
    #     return results,end-start
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
        
    # def generate_initial_condition(self,ol_list,ou_list):
    #         ul_p=self.ub_x[0]
    #         ll_p=self.lb_x[0]
    #         x0=self.rng.uniform(ll_p,ul_p,2)
    #         ol_list_cat_x=np.concatenate((ol_list[0],np.array([ul_p])))
    #         ol_list_cat_y=np.concatenate((ol_list[1],np.array([ul_p])))
    #         ou_list_cat_x=np.concatenate((ou_list[0],np.array([ll_p])))
    #         ou_list_cat_y=np.concatenate((ou_list[1],np.array([ll_p])))
    #         for i in range(ol_list.shape[1]):
    #             if (int(x0[0]>ol_list[0][i])+int(x0[1]>ol_list[1][i])+int(x0[0]<ou_list[0][i])+int(x0[1]<ou_list[1][i]))>(2*2-1):
    #                 diff_x=[abs(ol_list[0][i]-x0[0]), abs(ou_list[0][i]-x0[0])]
    #                 diff_y=[abs(ol_list[1][i]-x0[1]), abs(ou_list[1][i]-x0[1])]
    #                 diff_vector=diff_x+diff_y
    #                 n_sides=self.rng.integers(low=0,high=2)
    #                 minimum_distance=min(diff_vector)
    #                 minimum_indices=[]
    #                 minimum_indices.append(diff_vector.index(minimum_distance))
    #                 if n_sides>0:
    #                     if minimum_distance in diff_x:
    #                         minimum_indices.append(diff_vector.index(min(diff_y)))
    #                     else:
    #                         minimum_indices.append(diff_vector.index(min(diff_x)))

    #                 if(0 in minimum_indices):
    #                     l=max([x for x in list(ou_list_cat_x) if ol_list[0][i]>=x])
    #                     x0[0]=self.rng.uniform(l,ol_list[0][i])
    #                 if(2 in minimum_indices):
    #                     l=max([y for y in list(ou_list_cat_y) if ol_list[1][i]>=y])
    #                     x0[1]=self.rng.uniform(l,ol_list[1][i])
    #                 if(1 in minimum_indices):
    #                     u=min([x for x in list(ol_list_cat_x) if ou_list[0][i]<=x])
    #                     x0[0]=self.rng.uniform(ou_list[0][i],u)
    #                 if(3 in minimum_indices):
    #                     u=min([y for y in list(ol_list_cat_y) if ou_list[1][i]<=y])
    #                     x0[1]=self.rng.uniform(ou_list[1][i],u)
    #         return list(x0)+[0,0]