from logging import NullHandler
from lzma import MODE_NORMAL
from os import truncate
#from socket import SOL_ALG
#from tkinter import N
from typing import List

from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, quicksum
from itertools import chain
import cvxopt.modeling as cp
import itertools as itt
import numpy as np
import math
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Rectangle

@dataclass
class LP_SP_data:
    b    : np.ndarray
    lb   : np.ndarray
    ub   : np.ndarray

    x    : np.ndarray
    y    : np.ndarray
    lmbd : np.ndarray
    kappa: np.ndarray
    tau  : np.ndarray

    cost : float

    feas : bool

@dataclass()
class MILP_data:
    b      : np.ndarray

    opt_sp : LP_SP_data
    cover  : List[LP_SP_data]


    cov_set: set[tuple]  = field(init=False, repr=False) #[ [lb1,ub1],[lb2,ub2]... ]
    cov_set_train: np.ndarray = field(init=False, repr=False)

    x_opt  : np.ndarray         = field(init=False, repr=False) 
    y_opt  : np.ndarray         = field(init=False, repr=False)

    feas_nodes: List[LP_SP_data]= field(init=False, repr=False)


    def __post_init__(self):

        self.x_opt=self.opt_sp.x
        self.y_opt=self.opt_sp.y

        self.cov_set=[]
        self.feas_nodes=[]
        for sp in self.cover:
            # set(np.array([List(sp.lb.values()), List(sp.ub.values())])) ((2x1200))
            self.cov_set.append(tuple([tuple(list(sp.lb.values())), tuple(list(sp.ub.values()))]))
            
            if sp.feas!="infeasible":
                self.feas_nodes.append(sp)
        self.cov_set_train=np.asarray(self.cov_set)   
        self.cov_set=set(self.cov_set)
          
    
    def __eq__(self,other):

        return (self.b==other.b).all() and self.opt_sp==other.opt_sp and self.cov_set==other.cov_set and (self.x_opt==other.x_opt).all() and (self.y_opt==other.y_opt).all() and self.feas_nodes==other.feas_nodes



#EventHandler for collecting node information

class LPstatEventhdlr(Eventhdlr):
    """PySCIPOpt Event handler to collect data on LP events."""

    vars = {}
    
    def collectNodeInfo(self):#, firstlp=True):

        objval = self.model.getSolObjVal(None)

         # Collecting Solutions for all the variabeles and the binary variables for current Node
        LPsol_bin,x_sol,u_sol = {},{},{}
        
        if self.vars == {}:
            self.vars = self.model.getVars(transformed=False)
        
       
        for var in self.vars:
            solval = self.model.getSolVal(None, var)
            #LPsol_full[var.name]=self.model.getSolVal(None, var)
            if('x_' in var.name):
                x_sol[var.name]=self.model.getSolVal(None, var)
            if('u_' in var.name):
                u_sol[var.name]=self.model.getSolVal(None, var)
            if(var.vtype())=='BINARY':
                LPsol_bin[var.name] = self.model.getSolVal(None, var)

        node = self.model.getCurrentNode()
        vars=self.model.getVars()
        lb={}
        ub={}
        iters = self.model.lpiGetIterations()
        # Collecting lower bounds and upper bound of binary variables for current Node
        for v in vars:
            if v.vtype()=='BINARY':
                lb[v.name]=v.getLbLocal()
                ub[v.name]=v.getUbLocal()

        # Collecting leaves Nodes of the model at the current iteration

        openNodes=self.model.getOpenNodes()
        leavesNodesNumber=[]
        Open_nodes_lb={}
        for n in range(len(openNodes[0])):
            leavesNodesNumber.append(openNodes[0][n].getNumber())

        # Collecting Parents Ids
        if node.getNumber() != 1:
            parentnode = node.getParent()
            parent = parentnode.getNumber()
            
        else:
            parent = 0

        # Other iformations about the node
        depth = node.getDepth()
        cons=self.model.getConss()

        # Collecting dual solutions informations
        LPdual_sol={}
        for c in cons:
            LPdual_sol[c.name]=self.model.getDualSolVal(c)

        # Collecting information on the model and the solution boundaries

        age = self.model.getNNodes()

        pb = self.model.getPrimalbound()
        if pb >= self.model.infinity():
            pb = None

        nodedict = {
            "number": node.getNumber(),
            "type":node.getType(),
            "leavesNodesNumber":leavesNodesNumber,
            "LPsol_bin": LPsol_bin,
            #"LPsol_full":LPsol_full,
            "x_sol":x_sol,
            "u_sol":u_sol,
            "LPdual_sol":LPdual_sol,
            "objval": objval,
            #"first": firstlp,
            "parent": parent,
            "node":node,
            "age": age,
            "depth": depth,
            "lb":lb,
            "ub":ub,
            "constraints": cons,
            "primalbound": pb,
            "dualbound": self.model.getDualbound(),
        }
        self.nodelist.append(nodedict)
        # if firstlp:
        #     self.nodelist.append(nodedict)
        # elif iters > 0:
        #     prevevent = self.nodelist[-1]
        #     if nodedict["number"] == prevevent["number"] and not prevevent["first"]:
        #         # overwrite data from previous LP event
        #         self.nodelist[-1] = nodedict
        #     else:
        #         self.nodelist.append(nodedict)

    def eventexec(self, event):
        self.collectNodeInfo()    
        return {}
    
    # def eventexit(self):
    #     self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED,self)
    
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED,self)
    # def eventexec(self, event):
    #     if event.getType() == SCIP_EVENTTYPE.FIRSTLPSOLVED:
    #         self.collectNodeInfo(firstlp=True)
    #     elif event.getType() == SCIP_EVENTTYPE.LPSOLVED:
    #         self.collectNodeInfo(firstlp=False)
    #     else:
    #         print("unexpected event:" + str(event))
    #     return {}

    # def eventinit(self):
    #     self.model.catchEvent(SCIP_EVENTTYPE.LPEVENT, self)
    

class base_prob:

    def __init__(self):
        self.model=[]#Model()
        self.vars=vars()
        self.opt_vars=[]
        self.opt_vars_sp=[]
        self.LP_SP=LP_SP_Builder()
        
            
    def solve_(self):
        
        return{}
   
    def addVariable(self,dimensions,var_type,var_string,solve_type="full" ,lb=None, ub=None,LP_sol_bin=None):
        # dimensions= [size of variable per stage, number of stages, number of variables per stage]
        n_dim=len(dimensions)
        self.vars[var_string]={}
        x=self.vars[var_string]
        if solve_type=="full":
            m=self.model
        elif solve_type=="sp":
            m=self.LP_SP

              
        if n_dim==1:
            for i in range(dimensions[0]):
                x[i] = m.addVar( vtype=var_type,name=var_string+"(%s)"%(i),lb=lb,ub=ub)
        elif n_dim==2:
            for j in range(dimensions[1]):
                for i in range(dimensions[0]):                
                    x[i,j] = m.addVar( vtype=var_type,name=var_string+"(%s,%s)"%(i,j),lb=lb,ub=ub)
        elif n_dim==3:
            for i in range(dimensions[0]):
                for j in range(dimensions[1]):
                    for k in range(dimensions[2]):
                        var_name=var_string+"(%s,%s,%s)"%(i,j,k)
                        
                        if solve_type=="sp":
                            #try:
                            lb_bnd=lb[var_name]
                            ub_bnd=ub[var_name]
                            x[i,j,k] = m.addVar( vtype=var_type,name=var_string+"(%s,%s,%s)"%(i,j,k),lb=None,ub=None)
                            if (var_name=="bl(%s,%s,%s)"%(i,j,k) or var_name=="bu(%s,%s,%s)"%(i,j,k)): 
                                    # Cases :  b_opt=={0,1},                0<b_opt<1
                                    #     lb_bnd=ub_bnd, lb_bnd<ub_bnd      lb_bnd<ub_bnd
                                 if(LP_sol_bin==None):
                                    m.addCons(lb_bnd<=x[i,j,k], name="c_lb"+var_name)
                                    m.addCons(ub_bnd>=x[i,j,k],name="c_ub"+var_name)
                                 else:    
                                    if (LP_sol_bin[var_name] not in [0.0,1.0]) or (lb_bnd==ub_bnd): 
                                        m.addCons(lb_bnd<=x[i,j,k], name="c_lb"+var_name)
                                        m.addCons(ub_bnd>=x[i,j,k],name="c_ub"+var_name)
                                    else:
                                        # 0<=b         vs        0>=b, b>=0
                                        # lmbda*(-b)=0,     lmbda_m(b)=0, lmbda_p(-b)=0
                                        # in L(lmbda), 
                                        # lmbda*(-b)       (lmbda_p-lmbda_m)(-b)
                        
                                        #lmbda*=lmbda_m
                                        if LP_sol_bin[var_name]==1.0:
                                            m.addCons(x[i,j,k]>=lb_bnd,name="c_lb"+var_name)
                                            #m.addCons(x[i,j,k]>=LP_sol_bin[var_name],name="c_ubg="+var_name)
                                            m.addCons(x[i,j,k]<=LP_sol_bin[var_name],name="c_lb="+var_name)
                                            m.addCons(x[i,j,k]>=ub_bnd,name="c_ub"+var_name)

                                        else:
                                            m.addCons(lb_bnd>=x[i,j,k],name="c_lb"+var_name)
                                            m.addCons(LP_sol_bin[var_name]<=x[i,j,k],name="c_ub="+var_name)
                                            #m.addCons(x[i,j,k]<=LP_sol_bin[var_name],name="c_lbl="+var_name)
                                            m.addCons(x[i,j,k]<=ub_bnd,name="c_ub"+var_name)
                                
                                # if(LP_sol_bin!=None):
                                #     if (LP_sol_bin[var_name] in [0.0,1.0]) and (lb_bnd!=ub_bnd):
                                #         lb_bnd=LP_sol_bin[var_name]
                                #         ub_bnd=LP_sol_bin[var_name]  
                                # m.addCons(float(lb_bnd)<=x[i,j,k],name="c_lb"+var_name)
                                # m.addCons(float(ub_bnd)>=x[i,j,k],name="c_ub"+var_name)
                            #except:
                            #    raise Exception("Bounds only supported for binary variables")
                        else:                             
                            x[i,j,k] = m.addVar( vtype=var_type,name=var_string+"(%s,%s,%s)"%(i,j,k),lb=lb,ub=ub)
                
       
        else:
            raise Exception("More than 3 dimensions not supported currently")
        
        if solve_type=="full":
            self.opt_vars.append(self.vars[var_string])
        elif solve_type=="sp":
            self.opt_vars_sp.append(self.vars[var_string])


        

        
class OA_MIMPC(base_prob):
    def __init__(self,Q = 100,
                R = 50, N_obs_max=10, N=20, 
                lb_x=np.array([-10,-10,-8,-8]),
                ub_x=-np.array([-10,-10,-8,-8]),
                lb_u=np.array([-5,-5]),
                ub_u=-np.array([-5,-5])):

        super(OA_MIMPC,self).__init__()
        self.N_obs_max=N_obs_max
        self.lb_x=lb_x
        self.lb_u=lb_u
        self.ub_x=ub_x
        self.ub_u=ub_u
        self.N=N
        self.A = np.matrix([[1, 0, 0.1, 0],[0, 1, 0, 0.1],[0, 0, 1, 0],[0, 0, 0, 1]])
        self.B = 0.1*np.matrix([[0, 0],[0, 0],[1, 0],[0, 1]])
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
                
    def add_variables(self,solve_type="full", lb=None, ub=None, LP_sol_bin=None):
        self.free_x=[]
        self.free_cost=0
        if solve_type=='sp':
            self.opt_vars_sp=[]
            self.LP_SP.reset()
        elif solve_type=='full':
            self.model=Model()
            self.opt_vars=[]
            self.addVariable([self.N], 'C','tx',solve_type=solve_type,LP_sol_bin=LP_sol_bin)
            self.addVariable([self.N], 'C','tu',solve_type=solve_type,LP_sol_bin=LP_sol_bin)

        self.addVariable([self.nx, self.N+1], 'C','x_',solve_type=solve_type,LP_sol_bin=LP_sol_bin)
        self.addVariable([self.nu, self.N], 'C','u_',solve_type=solve_type,LP_sol_bin=LP_sol_bin)
        
        self.addVariable([self.space_dim, self.N, self.N_obs_max],lb=lb, ub=ub, var_type='B',var_string='bl',solve_type=solve_type,LP_sol_bin=LP_sol_bin)
        self.addVariable([self.space_dim, self.N, self.N_obs_max],lb=lb, ub=ub, var_type='B',var_string='bu',solve_type=solve_type,LP_sol_bin=LP_sol_bin)

        
        
                  
    def add_cost(self,solve_type="full"):
        Q=np.squeeze(np.asarray(self.Q))
        for i in range(self.N):
            self.free_cost+=np.linalg.norm((1+100*(i==self.N-1))*np.dot(Q,self.free_x[i,:]),np.inf)
     
        cost=0
        if solve_type=="full":
            m=self.model
            tx,tu,x,u,bl,bu=self.opt_vars
            for k in range(self.N):
                cost+=tx[k]+tu[k]
        elif solve_type=="sp":
            m=self.LP_SP
            x,u,bl,bu=self.opt_vars_sp
            a=1
            for k in range(self.N):
                # want to compute ||Q@x[:,k]||_inf=max(abs(Q@x[:,k])) 
                # abs(Q@x[:,k])=|[sum_j Q[0,j]*x[j,k], sum_j Q[1,j]*x[j,k], ... ,sum_j Q[n_x-1,j]*x[j,k]]|   
                # cost+=max(abs(cp.sum(a*self.Q[i,j]*x[j,k] for j in range(self.nx))))+max(abs(cp.sum(self.R[i,j]*u[j,k] for j in range(self.nu))))    
                sum_x=[abs(cp.sum([(1+100*(k==self.N-1))*float(self.Q[i,j])*x[j,k+1] for j in range(self.nx)])) for i in range(self.nx)]
                sum_u=[abs(cp.sum([float(self.R[i,j])*u[j,k] for j in range(self.nu)])) for i in range(self.nu)]
                cost+=cp.max(sum_x)+cp.max(sum_u) 

                    
        m.setObjective(cost,"minimize")
        
    def add_system_constraints(self, x0,solve_type="full"):
        self.x0=x0
        x_k=x0
        A=np.squeeze(np.asarray(self.A))
        self.free_x.append(x_k)
        for k in range(self.N):
            x_k=np.dot(A,x_k)
            self.free_x.append(x_k)    
        self.free_x=np.asarray(self.free_x)

        

        if solve_type=='full':
            m=self.model
            tx,tu,x,u,bl,bu=self.opt_vars
        elif solve_type=='sp':
            m=self.LP_SP
            x,u,bl,bu=self.opt_vars_sp
        
   
        for k in range(self.N+1):
            for i in range(self.nx):
                if k==0:
                    m.addCons(x[i,k]==float(x0[i]), name="c_init_x0(%s)"%i)
                else:
                    
                    if solve_type=="full": 
                        m.addCons(x[i,k]==quicksum(self.A[i,j]*x[j,k-1] for j in range(self.nx))\
                                            +quicksum(self.B[i,j]*u[j,k-1] for j in range(self.nu)), name="c_dyn_(%s,%s)"%(i,k-1))
                        m.addCons((1+100*(k==self.N))*self.Q[i,i]*x[i,k]<=tx[k-1], name="c_cost_+_(%s)_tx(%s)"%(i,k-1))
                        m.addCons(-(1+100*(k==self.N))*self.Q[i,i]*x[i,k]<=tx[k-1], name="c_cost_-_(%s)_tx(%s)"%(i,k-1))
                        if i<self.nu:
                            m.addCons(self.R[i,i]*u[i,k-1]<=tu[k-1], name="c_cost_+_(%s)_tu(%s)"%(i,k-1))
                            m.addCons(-self.R[i,i]*u[i,k-1]<=tu[k-1], name="c_cost_-_(%s)_tu(%s)"%(i,k-1))  
                    elif solve_type=='sp':
                        m.addCons(x[i,k]==cp.sum([float(self.A[i,j])*x[j,k-1] for j in range(self.nx)])\
                                            +cp.sum([float(self.B[i,j])*u[j,k-1] for j in range(self.nu)]), name="c_dyn_(%s,%s)"%(i,k-1))                        
                          
                    m.addCons(x[i,k]<=float(self.ub_x[i]), name="c_xub_(%s,%s)"%(i,k))
                    m.addCons(x[i,k]>=float(self.lb_x[i]), name="c_xlb_(%s,%s)"%(i,k))
                    if i<self.nu:#assuming nu<nx
                        m.addCons(u[i,k-1]<=float(self.ub_u[i]), name="c_uub_(%s,%s)"%(i,k-1))
                        m.addCons(u[i,k-1]>=float(self.lb_u[i]), name="c_ulb_(%s,%s)"%(i,k-1))  

    def add_OA_constraints(self, ol=np.array([[1,1,3,9,10],[3,-1,2,5,5]]),obs_size=np.array([[1,1,1,1,2],[2,2,2,2,1]]),solve_type="full"):

        M=self.M     
       
        self.ol=ol
        self.obs_size=obs_size
        self.ou =self.ol+self.obs_size
         
        if solve_type=='full':
            m=self.model
            tx,tu,x,u,bl,bu=self.opt_vars
        elif solve_type =='sp':
            m=self.LP_SP
            x,u,bl,bu=self.opt_vars_sp

        for k in range(self.N):
            for i in range(self.space_dim):
                for l in range(self.N_obs_max):
                    m.addCons(x[i,k+1]<=float(self.ol[i,l])+float(M)*bl[i,k,l], name="c_oa_l_(%s,%s,%s)"%(i,k,l))
                    
                    m.addCons(x[i,k+1]>=float(self.ou[i,l])-float(M)*bu[i,k,l], name="c_oa_u_(%s,%s,%s)"%(i,k,l))
                   
                    
                    if i==0:
                        m.addCons(bl[0,k,l]+bl[1,k,l]+bu[0,k,l]+bu[1,k,l]<=2*2-1, name="c_oa_(%s,%s)"%(k,l))
                        
        
     

    def retreive_sol(self):
        binary_sol={}
        b_l_sol, b_u_sol=[],[]
        lambda_sol={}
        x_sol=[self.x0]
        u_sol=[]
        return_dict={}
    
        m=self.model
        tx,tu,x,u,bl,bu=self.opt_vars
        
      



        Variables=m.getVars(transformed=False)

            
        return_dict.update({"lambda_sol":lambda_sol})
            
        
        if(m.getStatus()=="infeasible"): 
            status="infeasible"
            return_dict.update({"status":status})           
        else:
            status="optimal"
            return_dict.update({"status":status})
            objval = m.getPrimalbound()
            return_dict.update({"optimal_cost":objval})
            for i in range(self.N):
                x_sol.append([])
                u_sol.append([])
                for j in range(self.nx):
                    x_sol[i+1].append(m.getVal(x[j,i+1]))
                for j in range (self.nu):
                    u_sol[i].append(m.getVal(u[j,i]))
            return_dict.update({"x_sol":np.array(x_sol)})
            return_dict.update({"u_sol":np.array(u_sol)})
            for i in range(self.space_dim):
                b_l_sol.append([])
                b_u_sol.append([])
                for j in range(self.N):
                    b_l_sol[i].append([])
                    b_u_sol[i].append([])
                    for k in range(self.N_obs_max):
                        b_l_sol[i][j].append(m.getVal(bl[i,j,k]))    
                        b_u_sol[i][j].append(m.getVal(bu[i,j,k]))
                        binary_sol[bl[i,j,k].name]=m.getVal(bl[i,j,k])
                        
            for i in range(self.space_dim):
                for j in range(self.N):
                    for k in range(self.N_obs_max):
                        binary_sol[bu[i,j,k].name]=m.getVal(bu[i,j,k])


            return_dict.update({"b_l_sol":b_l_sol})
            return_dict.update({"b_u_sol":b_u_sol})
            return_dict.update({"binary_sol":binary_sol})
           
        
        return return_dict  

    def retreive_sol_cvxopt(self):

    
        m=self.LP_SP
        x,u,bl,bu=self.opt_vars_sp
        return_dict={}
        binary_sol={}
        LPsol_full={}
        Variables=m.prob.variables()
        lambda_vals=[float(m.constraint_dict[c].multiplier.value[0]) for c in m.constraint_dict.keys() if "=" not in c]
        # return_dict.update({"lambda_sol":np.array([float(c.multiplier.value[0]) for c in m.constraintList])})
        return_dict.update({"lambda_sol":np.array(lambda_vals)})
        objval=m.prob.objective.value()    
                
                

        if(objval==None): 
            status="infeasible"
          
            return_dict.update({"status":status})      
            return_dict.update({"optimal_cost":objval})
            return_dict.update({"x_sol":self.free_x.reshape((-1,1))})
            return_dict.update({"u_sol":self.free_u.reshape((-1,1))})
            return_dict.update({"binary_sol":np.zeros(2*(len(bl),1))})
            return_dict.update({"optimal_cost":self.free_cost})
            
        else:
            status="optimal"
            return_dict.update({"status":status})
            #objval = m.getPrimalbound()
            objval=float(m.prob.objective.value()[0])
            return_dict.update({"optimal_cost":objval})
            return_dict.update({"x_sol":np.array([float(x_i.value[0]) for x_i in x.values()])})
            return_dict.update({"u_sol":np.array([float(u_i.value[0]) for u_i in u.values()])})
            return_dict.update({"b_l_sol":np.array([float(bl_i.value[0]) for bl_i in bl.values()])})
            return_dict.update({"b_u_sol":np.array([float(bu_i.value[0]) for bu_i in bu.values()])})
            
            return_dict.update({"x_sol_dict":{xi.name:float(xi.value[0]) for xi in x.values()}})
            return_dict.update({"u_sol_dict":{ui.name:float(ui.value[0]) for ui in u.values()}})

             
                        
            for i in range(self.space_dim):
                for j in range(self.N):
                    for k in range(self.N_obs_max):
                        binary_sol[bu[i,j,k].name]=float(bu[i,j,k].value[0])
                        binary_sol[bl[i,j,k].name]=float(bl[i,j,k].value[0])


          
            return_dict.update({"binary_sol":binary_sol})
            for var in Variables:
                LPsol_full[var.name]=float(var.value[0])
            
            return_dict.update({"LPsol_full":LPsol_full})
        



        return return_dict

    def solve_(self, x0=np.array([10, 6.5,0,0]), 
                    ol=np.array([[1,1,3,9,10],[3,-1,2,5,5]]),
                    obs_size=np.array([[1,1,1,1,2],[2,2,2,2,1]]),
                    solve_type='full',
                    lb=None, ub=None,LP_sol_bin=None):
        
            
        self.M=2*self.ub_x[0]
        self.add_variables(solve_type=solve_type,lb=lb, ub=ub, LP_sol_bin=LP_sol_bin)
        self.add_system_constraints(x0,solve_type=solve_type)
        self.add_OA_constraints(ol=ol,obs_size=obs_size,solve_type=solve_type)
        self.add_cost(solve_type=solve_type) 

        if solve_type=="full":
     
            m=self.model
            self.nodelist=[]
            eventhdlr = LPstatEventhdlr()
            eventhdlr.nodelist = self.nodelist
        
            m.disablePropagation()
            m.setSeparating(3)
            m.setPresolve(3)
            m.setHeuristics(3)


            m.setParam("propagating/vbounds/freq","0")
            #m.setParam("misc/usesymmetry","0")
            m.setParam("lp/disablecutoff","1")
            #m.setParam("conflict/enable","0")
            #m.setParam("heuristics/feaspump/freq","20")
            #m.setParam("estimation/restarts/restartpolicy","n")
            m.setParam("heuristics/actconsdiving/freq","20")
            #m.setParam("heuristics/adaptivediving/freq","5")
            #m.setParam("heuristics/clique/freq","10")
            #m.setParam("heuristics/completesol/freq","10")
            #m.setParam("heuristics/conflictdiving/freq","10")
            #m.setParam("heuristics/distributiondiving/freq","10")
            #m.setParam("heuristics/fracdiving/freq","10")
            #m.setParam("heuristics/crossover/freq","30")

            #m.setParam("heuristics/farkasdiving/freq","10")
            #m.setParam("heuristics/rounding/freq","1")
            #m.setParam("numerics/feastol","1e-10")
            m.includeEventhdlr(
            eventhdlr, "LPstat", "generate LP statistics after every LP event"
                    )

            m.optimize()
            P_sol=self.retreive_sol()
        elif solve_type=="sp":
            self.LP_SP.solve__()                  
            P_sol=self.retreive_sol_cvxopt()
 
        return P_sol
       
    def getMILP_Data(self,parameters, MILP_covers, check_old_cov): #pass MILP_covers, check_old_cov
        x0_p=np.array(parameters["x0"])
        ol_p=parameters["ol"]
        obs_size_p=parameters["obs_size"]
        n_obs=ol_p.shape[1]
        # self.N_obs_max=n_obs
        if n_obs<self.N_obs_max:
            ol_p=np.concatenate((ol_p, self.ub_x[0]*np.ones((2,self.N_obs_max-n_obs))),axis=1)
            obs_size_p=np.concatenate((obs_size_p, np.zeros((2,self.N_obs_max-n_obs))),axis=1)

        P_sol=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p)
        b=np.vstack((x0_p.reshape((-1,1)),ol_p.reshape((-1,1)),obs_size_p.reshape((-1,1))))# qui va messo Psol=
        n_hndlr=nodelist_Handler(self.nodelist)
        self.n_hndlr=n_hndlr
        
        # retreive soln
        # Check if soln in leaves
        
        LP_SP_data_list=[]
        matched_sol={}
        difference_bin_sol={}
        difference_full_sol={}
        tmp_diff_x={}
        tmp_diff_u={}
        cost_diff={}
        differences=[]
        solution_found=False
        i=0
        min_diff_node=None
        for l in n_hndlr.leaves:#leaves_solution:
            if abs(l["objval"]-P_sol["optimal_cost"])<1e-5:
                diff=0.0
                for var in P_sol["binary_sol"].keys():
                    diff=max([diff, abs(l['LPsol_bin'][var]-P_sol["binary_sol"][var])])
                differences.append(diff)
                if differences[i]==min(differences):
                    min_diff_node=l
                i=i+1
        if bool(min_diff_node):
            min_diff_node['LPsol_bin']=P_sol["binary_sol"]
            solution_node=min_diff_node
            solution_found=True
        else:
            # self.epsilon=np.linalg.norm(np.array([list(min_diff_node['LPsol_bin'].values())]).reshape((-1,1))-np.array([list(P_sol["binary_sol"].values())]).reshape((-1,1)),np.inf)
            # self.epsilon_list.append(self.epsilon)
            # solution_node=min_diff_node
            # solution_found=True
            # if any(epsilon_i>self.epsilon_max for epsilon_i in self.epsilon_list):
            print("MILP was not solved!")
                

        
                       
        # if not solution_found:
        #     while (not flag):
        #         if(np.linalg.norm(np.array([list(l['LPsol_bin'].values())])-np.array([list(P_sol["binary_sol"].values())]),1)<=self.epsilon):
        #             l['LPsol_bin']=P_sol["binary_sol"]
        #             solution_node=l
        #             solution_found=True
        #             flag=True
        #         else:
        #             self.epsilon=np.linalg.norm(np.array([list(l['LPsol_bin'].values())])-np.array([list(P_sol["binary_sol"].values())]),1)
        #             self.epsilon_list.append(self.epsilon)
        #             if any(epsilon_i>self.epsilon_max for epsilon_i in self.epsilon_list):
        #                 print("Warning node approximation of the Binary solution is Bad")
        
        sol_lb=solution_node["lb"]; sol_ub=solution_node["ub"]
        bin_sol=P_sol["binary_sol"]
        binary_vars=list(sol_lb.keys())
        sol_sp=tuple([tuple(list(sol_lb.values())), tuple(list(sol_ub.values()))])
        new_cov=None
        for cov in MILP_covers:
            if sol_sp in cov and check_old_cov:
                new_cov=cov
    
        if new_cov is not None and solution_found:  
            for sp in new_cov:
                LP_lb={}
                LP_ub={}
                lb=sp[0]
                ub=sp[0]
                for i in range(len(binary_vars)):
                    LP_lb.update({binary_vars[i]:lb[i]})
                    LP_ub.update({binary_vars[i]:ub[i]})
                if sp==sol_sp:
                    P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp", LP_sol_bin=bin_sol)
                else:
                    P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp")
                    

                if P_sol_lp["status"]!="infeasible":
                    LP_SP_data_point=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                        x=np.vstack((P_sol_lp["x_sol"].reshape((-1,1)),P_sol_lp["u_sol"].reshape((-1,1)))),
                        y=np.array(list(P_sol_lp["binary_sol"].values())),lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),#np.array(list(.values())),
                        kappa=0.0, tau=1.0,cost=P_sol_lp["optimal_cost"],feas=P_sol_lp["status"])
                else:
                    objval=self.free_cost #from free evolution
                    dualcost=1.0
                    LP_SP_data_point=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                                                x=np.vstack((self.free_x.reshape((-1,1)),self.free_u.reshape((-1,1)))), #change to free evolution state-input trajectory
                                                y=np.zeros((len(LP_lb.values()),1)), #just set to all binry variables
                                                lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),
                                                kappa=objval-dualcost, tau=0.0,
                                                cost=objval,feas=P_sol_lp["status"])

                LP_SP_data_list.append(LP_SP_data_point)
        

        

        elif new_cov is None:
            #make a Cover
            r_list=n_hndlr.make_Cover()
            # r_list=n_hndlr.make_min_Cover()
            #Modifing the minimum depth Node to construct a cover
            mod_node=None
            if not r_list[0]:
                if len(n_hndlr.leaves)>1:
                    mod_node_idx=r_list[1]
                    mod_node=n_hndlr.leaves[mod_node_idx]
                    LP_lb=mod_node["lb"]
                    LP_ub=mod_node["ub"]

                    P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp")# LP_sol_bin=LP_sol_bin)
                    
                    if P_sol_lp["status"]!="infeasible":
                    
                        n_hndlr.leaves[mod_node_idx]["LPsol_bin"]=P_sol_lp["binary_sol"]
                        if len(self.n_hndlr.infeasibleLeavesId)>0:
                            n_hndlr.infeasibleLeavesId=n_hndlr.infeasibleLeavesId-set([mod_node["number"]])
                            n_hndlr.feasibleLeavesId=n_hndlr.feasibleLeavesId|set([mod_node["number"]])
                        
                        LP_SP_data_mod=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                            x=np.vstack((P_sol_lp["x_sol"].reshape((-1,1)),P_sol_lp["u_sol"].reshape((-1,1)))),
                            y=np.array(list(P_sol_lp["binary_sol"].values())),lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),#np.array(list(.values())),
                            kappa=0.0, tau=1.0, cost=P_sol_lp["optimal_cost"],feas=P_sol_lp["status"])
                        
                    else:
                        objval=self.free_cost#from free evolution
                        dualcost=1.0
                        LP_SP_data_mod=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                                                    x=np.vstack((self.free_x.reshape((-1,1)),self.free_u.reshape((-1,1)))), #change to free evolution state-input trajectory
                                                    y=np.zeros((len(LP_lb.values()),1)), #just set to all binry variables
                                                    lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),
                                                    kappa=objval-dualcost, tau=0.0,
                                                    cost=objval,feas=P_sol_lp["status"])
                else:
                    uncvrd_vars=r_list[2]
                    lbtilde=r_list[3]
                    ubtilde=r_list[4]
                    add_node=n_hndlr.leaves[0]
                    LP_lb=add_node["lb"].copy()
                    LP_ub=add_node["ub"].copy()
                    for var in uncvrd_vars:
                        if lbtilde[var]==1.0:
                            LP_lb[var]=0.0
                            LP_ub[var]=0.0
                        elif ubtilde[var]==0.0:
                            LP_lb[var]=1.0
                            LP_ub[var]=1.0

                    P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp")# LP_sol_bin=LP_sol_bin)
                    if P_sol_lp["status"]!="infeasible":
                        objval=P_sol_lp["optimal_cost"]
                        dualval=objval
                        binsol=P_sol_lp["binary_sol"]
                        xsol=P_sol_lp["x_sol"]
                        usol=P_sol_lp["u_sol"]
                    else:
                        objval=1e20
                        dualval=0.0
                        binsol=np.zeros((len(LP_lb.values()),1))
                        xsol=0.0
                        usol=0.0
                    nodedict = {
                                "number": -1,
                                "type":0,
                                "leavesNodesNumber":0,
                                "LPsol_bin": binsol,
                                #"LPsol_full":LPsol_full,
                                "x_sol":xsol,
                                "u_sol":usol,
                                "LPdual_sol":P_sol_lp["lambda_sol"],
                                "objval": objval,
                                #"first": firstlp,
                                "parent": 1,
                                "node":None,
                                "age": add_node["age"],
                                "depth": add_node["depth"],
                                "lb":LP_lb,
                                "ub":LP_ub,
                                "constraints": add_node["constraints"],
                                "primalbound": objval,
                                "dualbound": dualval}
                    n_hndlr.nodelist.append(nodedict)
                    n_hndlr.leaves.append(nodedict)
                    mod_node=n_hndlr.leaves[1]
                    n_hndlr.nodesId=n_hndlr.nodesId|set([nodedict["number"]])

                    
                    
                    if P_sol_lp["status"]!="infeasible":
                    
                        n_hndlr.feasibleLeavesId=n_hndlr.feasibleLeavesId|set([nodedict["number"]])
                        LP_SP_data_mod=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                            x=np.vstack((P_sol_lp["x_sol"].reshape((-1,1)),P_sol_lp["u_sol"].reshape((-1,1)))),
                            y=np.array(list(P_sol_lp["binary_sol"].values())),lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),#np.array(list(.values())),
                            kappa=0.0, tau=1.0,cost=P_sol_lp["optimal_cost"],feas=P_sol_lp["status"])
                        
                    else:
                        objval=self.free_cost
                        dualcost=1.0
                        n_hndlr.infeasibleLeavesId=n_hndlr.infeasibleLeavesId|set([nodedict["number"]])
                        LP_SP_data_mod=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                                                    x=np.vstack((self.free_x.reshape((-1,1)),self.free_u.reshape((-1,1)))),
                                                    y=binsol, lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),
                                                    kappa=objval-dualcost, tau=0.0,
                                                    cost=objval,feas=P_sol_lp["status"])


                    
                    

        r_list=n_hndlr.make_min_Cover()
        # solving again all the nodes corresponding to feasible leaves to get lambdas       
        for l in n_hndlr.leaves:
            LP_lb=l["lb"]
            LP_ub=l["ub"]
            if  len(n_hndlr.leaves)==1:
                LP_sol_bin=l["LPsol_bin"]
                x_sol_l=l["x_sol"]
                u_sol_l=l["u_sol"]
                
                P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp", LP_sol_bin=LP_sol_bin)
                
                LP_SP_data_point=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                    x=np.vstack((P_sol_lp["x_sol"].reshape((-1,1)),P_sol_lp["u_sol"].reshape((-1,1)))),
                    y=np.array(list(P_sol_lp["binary_sol"].values())),lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),#np.array(list(.values())),
                    kappa=0.0, tau=1.0,cost=P_sol_lp["optimal_cost"],feas=P_sol_lp["status"])
                opt_sp=LP_SP_data_point
                LP_SP_data_list.append(LP_SP_data_point)
            elif l!=mod_node:
                if(l["number"] in n_hndlr.feasibleLeavesId):
                    LP_sol_bin=l["LPsol_bin"]
                    x_sol_l=l["x_sol"]
                    u_sol_l=l["u_sol"]
                    objval=self.free_cost
                    dualcost=1
                    P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp", LP_sol_bin=LP_sol_bin)
                    
                    LP_SP_data_point=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                        x=np.vstack((P_sol_lp["x_sol"].reshape((-1,1)),P_sol_lp["u_sol"].reshape((-1,1)))),
                        y=np.array(list(P_sol_lp["binary_sol"].values())),lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),#np.array(list(.values())),
                        kappa=0.0+(objval-dualcost)*(P_sol_lp["status"]=="infeasible"), tau=1.0-(P_sol_lp["status"]=="infeasible"),cost=P_sol_lp["optimal_cost"],feas=P_sol_lp["status"])
                    
                    ### This is for debuging we check if the binary solution given by cvxpy 
                    ### match that proposed from the SCIP solver at  the specified leave
                    ### otherwhise we compute the different between the two solutions
                    
                    if(P_sol_lp["binary_sol"]==LP_sol_bin):
                        matched_sol[l["number"]]=LP_sol_bin
                    else:
                        difference_bin_sol[l["number"]]=np.array(list(P_sol_lp["binary_sol"].values()))-np.array(list(LP_sol_bin.values()))
                        for k in x_sol_l.keys():
                            tmp_diff_x[k]=P_sol_lp["x_sol_dict"][k]-x_sol_l[k]
                        for k in u_sol_l.keys():
                            tmp_diff_u[k]=P_sol_lp["u_sol_dict"][k]-u_sol_l[k]
                        difference_full_sol[l["number"]]=np.sqrt(np.sum(np.square(np.concatenate((np.array(list(tmp_diff_x.values())),np.array((list(tmp_diff_u.values()))))))))
                        cost_diff[l["number"]]=P_sol_lp["optimal_cost"]-l["objval"]
                    ###
                    if(l==solution_node):
                        opt_sp=LP_SP_data_point
                    ###
                    
                else:
                    
                    P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp")
                    objval=self.free_cost
                    dualcost=1.0
                    LP_SP_data_point=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                        x=np.vstack((self.free_x.reshape((-1,1)),self.free_u.reshape((-1,1)))),
                        y=np.zeros((len(LP_lb.values()),1)), lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),
                        kappa=objval-dualcost, tau=0.0,cost=objval,feas=P_sol_lp['status'])
                LP_SP_data_list.append(LP_SP_data_point)
        try:
            LP_SP_data_list.append(LP_SP_data_mod)
        except:
            LP_SP_data_list=LP_SP_data_list
        try:
            opt_sp=opt_sp
        except:
            print("warning")

        return MILP_data(b=b,opt_sp=opt_sp,cover=LP_SP_data_list)#,matched_sol,difference_full_sol,cost_diff
        
#### Functions for random parameter generation

    def sample_params(self, rng):

        # rng=np.random.default_rng(seed)

        def generate_random_obstacles( N_obs_max,target_margin,upper_limit, lower_limit, rng):
            ol_list=[[],[]]
            obs_size_list=[[],[]]
            obstacle_limits=[[],[]]
          
            
            n_obs=rng.integers(1,N_obs_max,1)
            obs_max_size=max([(upper_limit-lower_limit)/(n_obs)-(upper_limit+target_margin)*(n_obs==1)-(target_margin)*(n_obs==2),upper_limit/2-target_margin])
            
      
            
            for i in range(list(n_obs)[0]):
                ol_x=float(rng.uniform(lower_limit,upper_limit-obs_max_size))#[lower_bound ,upper_bound-maxobs_size]
                ol_x=round(ol_x,3)
                if(ol_x>-obs_max_size-target_margin and ol_x<target_margin):
                    ol_y=[float(rng.uniform(lower_limit, -obs_max_size-target_margin)), float(rng.uniform(target_margin, upper_limit-obs_max_size))]
                    ol_y= float(rng.choice(ol_y))
                    ol_y=round(ol_y,3)
                else:
                    ol_y=float(rng.uniform(lower_limit, upper_limit-obs_max_size))
                    ol_y=round(ol_y,3)
                obs_size=rng.uniform(0.5,obs_max_size,2)
                #obs_size=round(obs_size,3)

                ou_x=round(float(ol_x+round(obs_size[0],3)),3)
                ou_y=round(float(ol_y+round(obs_size[1],3)),3)

                
                
                if i==0:
                    ol_list[0].append((ol_x))
                    ol_list[1].append((ol_y))
                    obs_size_list[0].append(float(round(obs_size[0],3)))
                    obs_size_list[1].append(float(round(obs_size[1],3)))
        
                    obstacle_limits[0].append((float(ol_x),float(ou_x)))
                    obstacle_limits[1].append((float(ol_y),float(ou_y)))
                else:
                    union_x=[]
                    for begin,end in sorted(obstacle_limits[0]):
                        if union_x and union_x[-1][1]>=begin-1:
                            union_x[-1][1]=max(union_x[-1][1],end)
                        else:
                            union_x.append([begin,end])
                    union_y=[]
                    for begin,end in sorted(obstacle_limits[1]):
                        if union_y and union_y[-1][1]>=begin-1:
                            union_y[-1][1]=max(union_y[-1][1],end)
                        else:
                            union_y.append([begin,end])
                    if union_x != (lower_limit,upper_limit) and union_y !=(lower_limit,upper_limit):
                        ol_list[0].append(ol_x)
                        ol_list[1].append(ol_y)
                        obs_size_list[0].append(float(round(obs_size[0],3)))
                        obs_size_list[1].append(float(round(obs_size[1],3)))
        
                        obstacle_limits[0].append((float(ol_x),float(ou_x)))
                        obstacle_limits[1].append((float(ol_y),float(ou_y)))
                    else:
                        i=i-1
            return np.array(ol_list),np.array(obs_size_list),obstacle_limits

            
        def generate_initial_condition(obstacle_limits,upper_limit,lower_limit, rng):
            ul_p=upper_limit[0]
            ll_p=lower_limit[0]
            ul_v=upper_limit[2]

            def range_gaps(a, b, ranges):
                epsilon=0.005
                ranges = sorted(ranges)
                flat = chain((a,), chain.from_iterable(ranges), (b,))
                return [[x+epsilon, y-epsilon] for x, y in zip(flat, flat) if x+epsilon < y]
            
          
            available_spots_x=np.array(range_gaps(ll_p,ul_p,obstacle_limits[0]))
            
            available_spots_y=np.array(range_gaps(ll_p,ul_p,obstacle_limits[1]))
           
            random_peeks_x=[]
            random_peeks_y=[]
            for i in available_spots_x:
                while True:
                    peek_x=round(float(rng.uniform(i[0],i[1])),3)
                    if(peek_x>i[0] and peek_x<i[1]):
                        random_peeks_x.append(peek_x)
                        break
                    
            for j in available_spots_y:
                while True:
                    peek_y=round(float(rng.uniform(j[0],j[1])),3)
                    if(peek_y>j[0] and peek_y<j[1]):
                        random_peeks_y.append(peek_y)
                        break
            x0_x=rng.choice(random_peeks_x)
            x0_y=rng.choice(random_peeks_y)
            return  [x0_x,x0_y,0,0]#np.sign(-x0_x)*round(rng.uniform(0,1),3),np.sign(-x0_y)*round(rng.uniform(0,1),3)]#[0,0]

        target_margin=1    
        ol_list,obs_size_list,obstacle_limits=generate_random_obstacles(N_obs_max=self.N_obs_max,target_margin=target_margin,upper_limit=self.ub_x[0],lower_limit=self.lb_x[0], rng=rng)
        x0=generate_initial_condition(obstacle_limits, self.ub_x, self.lb_x, rng)
        # fig=plt.figure()
        # n_obs=obs_size_list.shape[-1]
        # plt.plot(x0[0], x0[1],'*')
        # for i in range(n_obs):
        #     plt.gca().add_patch(Rectangle((ol_list[0,i],ol_list[1,i]),obs_size_list[0,i],obs_size_list[1,i]))
        # plt.show()
        b={"x0":x0, "ol": ol_list, "obs_size": obs_size_list}
        return b


class nodelist_Handler:

    def __init__(self,nodelist):
        self.nodelist=nodelist
        self.parentsId=set(self.filter("parent"))
        self.nodesId=set(self.filter("number"))
        self.OpenNodesIdPerIteration=self.filter("leavesNodesNumber")
        self.allOpenNodesId=np.array([0])
        for i in range (len(self.OpenNodesIdPerIteration)):
            self.allOpenNodesId=np.append(self.allOpenNodesId,np.array(self.OpenNodesIdPerIteration[i]))
        self.allOpenNodesId=self.allOpenNodesId.astype(int)
        self.allOpenNodesId=set(self.allOpenNodesId)
        self.leavesId=self.nodesId-self.parentsId
        
        self.leaves=self.filter(None,None,"number",self.leavesId)
        x=[1e20]
        self.infeasibleLeavesId=set(self.filter("number",self.leaves,"objval",x))# here we have to modify because it is not always possible to define the infeasible leaves
        self.feasibleLeavesId=self.leavesId -self.infeasibleLeavesId
            
            # we have to modify infact is not always possible to find the leaves(if the problem
        #has just the root node you cannot define the leaves, in that case it does not make sense also the make cover infact the root node
        # will directly provide the cover and we just need to store it (or probably we can solve it again anyway))
        #
       
        self.binary_variablesId=list(self.nodelist[0]["LPsol_bin"].keys())
        self.leaves_sols=self.filter("LPsol_bin",self.leaves)
                               
    def filter(self,output_field=None,node_set=None,input_field=None,input_field_set=None):
        #The input set can be either a set of elements of the same fied as the input
        # or a set of nodes
        # Depending on the combination of input and output field we have different options
        if node_set==None:
            node_set=self.nodelist
        if input_field==None:
            if output_field!=None:
                filtered_nodelist=[l[output_field] for l in node_set ]
            else:
                raise Exception("At list one between Input Field and output filed must be specified")
        else:
            if(input_field_set==None):
                raise Exception("Input field set must be specified")
            if(type(list(input_field_set)[0])!=type(node_set[0][input_field])):
                raise Exception("Input field set doesn't match the elements in nodeset for the specified field")

            if output_field!=None and output_field!="index":
                filtered_nodelist=[l[output_field] for l in node_set if l[input_field] in input_field_set ]
            elif output_field=="index":
                filtered_nodelist=[node_set.index(l) for l in node_set if l[input_field] in input_field_set ]
            else:
                filtered_nodelist=[l for l in node_set if l[input_field] in input_field_set ]               

        return filtered_nodelist

    def make_min_Cover(self):
        if(bool(self.infeasibleLeavesId)):
            feasible_leaves=self.filter(node_set=self.leaves,input_field="number",input_field_set= self.feasibleLeavesId)
            infeasible_leaves=self.filter(node_set=self.leaves,input_field="number",input_field_set= self.infeasibleLeavesId)
            min_infeasible_leaves=[]
            return_list=self.make_Cover(feasible_leaves)
            uncovered_variables=return_list[2]
            lb_tilde=return_list[3]
            ub_tilde=return_list[4]
            
            for b in uncovered_variables:
                for l in infeasible_leaves:
                    if l["lb"][b]<lb_tilde[b] or l["ub"][b]>ub_tilde[b]:
                        if l not in min_infeasible_leaves:
                            min_infeasible_leaves.append(l)
                        continue

            new_leaves=feasible_leaves+min_infeasible_leaves
            self.leaves=new_leaves
            return_list=self.make_Cover()
            if return_list[0]:
                return_list.append(new_leaves)
                
            else:
                raise Exception("min_infeasible_leaves does not have modified node")
        else:
            return_list=self.make_Cover()
            if return_list[0]:
                return_list.append(self.leaves)
        
        return return_list

    def make_Cover(self, nodes=None):
        binary_vars=self.binary_variablesId
        Leaves_flag=False
        if nodes==None:
            nodes=self.leaves
            Leaves_flag=True

        lbs=self.filter("lb",nodes)
        ubs=self.filter("ub",nodes)
        upper_coverage=0
        lower_coverage=0
        uncovered_variables=[]
        lb_tilde={}
        ub_tilde={}
        for b in binary_vars:
            lb_tilde[b]=1
            ub_tilde[b]=0

            for l in range(len(nodes)):
                lb_tilde[b]=(min(lb_tilde[b],lbs[l][b]))
                ub_tilde[b]=(max(ub_tilde[b],ubs[l][b]))
        
            upper_coverage+=ub_tilde[b]
            lower_coverage+=lb_tilde[b]
            if ub_tilde[b]==lb_tilde[b]:
                uncovered_variables.append(b)
        


        if Leaves_flag and not(upper_coverage==len(binary_vars) and lower_coverage==0):
            # here we should contemplate the case that all the nodes are feasible in which case we need to modify the 
            # upper bound and lower bound of the feasible leaves with the worst value of the cost function
            # we have to take care because the make_cover is also called into the make_min_cover
            if bool(self.infeasibleLeavesId):
                infeasible_nodes=self.filter(None,nodes, "number",self.infeasibleLeavesId)
                infeasible_idx=self.filter("index",nodes, "number",self.infeasibleLeavesId)
                infeasible_depths=self.filter("depth",nodes, "number",self.infeasibleLeavesId)
                smallest_depth=min(infeasible_depths)
                # smallest_depth_infeasible_node=[l for l in infeasible_nodes if l["depth"]==smallest_depth ]
                smallest_depth_infeasible_node=infeasible_nodes[infeasible_depths.index(smallest_depth)]
                s_d_infeas_node_num=infeasible_idx[infeasible_depths.index(smallest_depth)]
                for s in uncovered_variables:
                    nodes[s_d_infeas_node_num]["ub"][s]=1.0
                    nodes[s_d_infeas_node_num]["lb"][s]=0.0
                    lb_tilde[s]=0.0
                    ub_tilde[s]=1.0
 
                return [False, s_d_infeas_node_num,uncovered_variables,lb_tilde,ub_tilde]
            else:
                if len(self.feasibleLeavesId)>1:
                    feasible_nodes=self.filter(None,nodes, "number",self.feasibleLeavesId)
                    feasible_idx=self.filter("index",nodes, "number",self.feasibleLeavesId)
                    feasible_obj=self.filter("objval",nodes, "number",self.feasibleLeavesId)
                    highest_obj=max(feasible_obj)
                    highest_obj_feasible_node=feasible_nodes[feasible_obj.index(highest_obj)]
                    ho_feas_node_num=feasible_idx[feasible_obj.index(highest_obj)]
                    for s in uncovered_variables:
                        nodes[ho_feas_node_num]["ub"][s]=1.0
                        nodes[ho_feas_node_num]["lb"][s]=0.0
                        lb_tilde[s]=0
                        ub_tilde[s]=1
                    

                    return [False, ho_feas_node_num,uncovered_variables,lb_tilde,ub_tilde]
                else:
                    return [False, 0, uncovered_variables,lb_tilde,ub_tilde]



        else:
            return [True,None,uncovered_variables,lb_tilde,ub_tilde]
                    
# Class that manage the dataset creation
class MILP_DataSet:

    def __init__(self,P, eps=5*1e-3, beta=5e-6, c=2*2**0.5+3**0.5 ):
        self.eps=eps
        self.beta=beta
        self.P=P
        self.c=c
        self.MILP_list=[]
        self.MILP_covers=[]
        self.cov_appearances=[]
        self.G=1e10
        self.N=0
        self.N1=0
        self.rng=np.random.default_rng(1)
        self.N_prop=50

    def add_MILP_problem(self, MILP_data):
        
        
        if MILP_data not in self.MILP_list: # the doubt here is that if compare two elements with same cover but different solution 
                                            #we add a new data point that is not needed moreover is more difficult to compare two MILP_data_points
                                            #because we have more field to check among which there is the cover field that is a collection of LP_SP_data points
                                            #each of which is caracterized by lb,ub, and solution for binary and continuos variables
                                            #and even if all the fields for the single subproblem have the same dimensionality we have different number of subproblem 
                                            # for each MILP data point because we have different number of leaves (both feasible and infeasible) so my suggestion is to check just 
                                        # the equality between cover sets, instead if we define the MILP_covers as a set there should be no problems
        
        #probably we have to compare just the set then the MILP will be added only if the cov_set is different from the already 
        #collected ones
            self.MILP_list.append(MILP_data)
            self.N+=1
            new_cover=MILP_data.cov_set
            if not new_cover in self.MILP_covers:
                self.MILP_covers.append(new_cover)
                self.cov_appearances.append(1)
                
            else:
                i=self.MILP_covers.index(new_cover)
                self.cov_appearances[i]=self.cov_appearances[i]+1
        
        self.N1=self.cov_appearances.count(1)
        self.G=float(self.N1/self.N)
    
    def check_prob_bnd(self):
        if self.G+self.c*np.sqrt(1/self.N*np.log(3/self.beta))<=self.eps:
            return True
        else:
            return False
    
    def sample_MILP_parameter(self):
        b=self.P.sample_params(self.rng)
        return b #dictionary

    def get_MILP_data(self, b):
        
        if len(self.MILP_covers)>50:
            check_covs=True
        else:
            check_covs=False
        MILP_data_point=self.P.getMILP_Data(b, self.MILP_covers, check_covs)

        return MILP_data_point
    
    def make_Dataset(self):
        param=self.sample_MILP_parameter()
        for k in range(self.N_prop):
            MILP_data_point=self.get_MILP_data(b=param)
            self.add_MILP_problem(MILP_data_point)
            u_opt=np.array([MILP_data_point.opt_sp.x[self.P.A.shape[0]*(self.P.N+1)],MILP_data_point.opt_sp.x[self.P.A.shape[0]*(self.P.N+1)+1]])
            x0_k=np.dot(self.P.A,np.array(param['x0']).reshape(self.P.nx,1))+np.dot(self.P.B,u_opt)
            param["x0"]=np.array(np.transpose(x0_k)).reshape(self.P.nx,).tolist()
        while not self.check_prob_bnd():
            param=self.sample_MILP_parameter()
            for k in range(self.N_prop):
                MILP_data_point=self.get_MILP_data(param)
                self.add_MILP_problem(MILP_data_point)
                u_opt=np.array([MILP_data_point.opt_sp.x[self.P.A.shape[0]*(self.P.N+1)],MILP_data_point.opt_sp.x[self.P.A.shape[0]*(self.P.N+1)+1]])
                x0_k=np.dot(self.P.A,np.array(param['x0']).reshape(self.P.nx,1))+np.dot(self.P.B,u_opt)
                param["x0"]=np.array(np.transpose(x0_k)).reshape(self.P.nx,).tolist()
            
class LP_SP_Builder:
    def __init__(self):
        self.constraintList = []
        self.str2constr = {}
        self.constraint_dict={}
        self.objective=[]
        self.prob=[]
    def solve__(self):
        self.prob=cp.op(self.objective,self.constraintList)
        self.prob.solve(solver='mosek')
        
    def reset(self):
        self.variableList=[]
        self.constraintList = []
        self.str2constr = {}
        self.constraint_dict={}
        self.objective=[]
        self.prob=[]
    
    def addVar(self,vtype,name,lb,ub):
        x=cp.variable(name=name)
        self.variableList.append(x)
        return x
    
        
    def setObjective(self,cost,criteria):

        if criteria=='minimize':
            self.objective=cost
        elif criteria=='maximize':
            self.objective==-cost

    def addCons(self, expr, name):
        self.constraintList.append(expr)
        self.str2constr[name] = len(self.constraintList)-1
        self.constraint_dict[name]=expr

    def getConstrList(self):
        return self.constraintList
    
    def getConstrDict(self):
        return self.constraint_dict

    def getConstr(self, str_):
        return self.constraintList[self.str2constr[str_]]




if __name__=="__main__":
    # x0=np.array([.5, 0.2,0,0])
    # ol=np.array([[1,1,3,9,10],[3,-1,2,5,5]])
    # obs_size=np.array([[1,1,1,1,2],[2,2,2,2,1]])
    # ou =ol+obs_size
    # fig=plt.figure()
    # n_obs=obs_size.shape[-1]
    # plt.plot(x0[0], x0[1],'*')
    # for i in range(n_obs):
    #     plt.gca().add_patch(Rectangle((ol[0,i],ol[1,i]),obs_size[0,i],obs_size[1,i]))
    # plt.show()
    N_obs_max=10
    P=OA_MIMPC(N_obs_max=N_obs_max)
    
    # parameters={}
    # parameters.update({"x0":x0})
    # parameters.update({"ol":ol})
    # parameters.update({"obs_size":obs_size})
    # [MILP_data_point,matched_sol,difference_full_sol,cost_diff]=P.getMILP_Data(parameters)
    Dataset_creator=MILP_DataSet(P=P)
    Dataset_creator.make_Dataset()
    Dataset=Dataset_creator.MILP_list
    Covers=Dataset_creator.MILP_covers
            
    # a=1

                



        

    


        


    

    
