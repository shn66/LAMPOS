
from pyscipopt import Model, quicksum
from utils import LP_SP_data,MILP_data,nodelist_Handler,LP_SP_Builder,LPstatEventhdlr
import cvxopt.modeling as cp
import numpy as np
import time
import _pickle as pkl
import pickletools
import gzip



class MILP_DataSet:
    '''
    Class that manage the dataset creation
    '''

    def __init__(self,P, eps=1*1e-1, beta=1e-3, c=2*2**0.5+3**0.5, save_data=False ):


        self.P=P
        self.c=c
        self.MILP_list=[]
        self.MILP_covers=[]
        self.cov_appearances=[]
        self.G=1e10                 #   Good Turing estimator
        self.N=0                    #   Total number of problems
        self.N1=0                   #   Number of problems oberved once
        self.eps=eps                #   Probability bound
        self.beta=beta              #   Confidence
        self.rng=np.random.default_rng(8)
        self.N_prop=5               #   Propagation horizon
        self.save=save_data


    def make_Dataset(self,data_set_file=None,MILPdata_file=None):
        '''
        Function for dataset construction 
        Input: Paths for saving datasets
        '''
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not data_set_file:
            data_set_file="/home/mpc/LMILP/Datasets/dataset_dt"+str(int(self.P.dt))+str(int(self.P.dt/0.1))+"_N"+str(self.P.N)+"_"+timestr+".p"
        if not MILPdata_file:
            MILPdata_file="/home/mpc/LMILP/Datasets/MILP_data_points_"+str(int(self.P.dt))+str(int(self.P.dt/0.1))+"_N"+str(self.P.N)+"_"+timestr+".p"
        
        with gzip.open(data_set_file, "wb") as f:       
            with gzip.open(MILPdata_file,"wb") as f1:  

                param=self.sample_MILP_parameter()  #   First Problem
                for k in range(self.N_prop):        #   Solution Propagation       
                    warm_start=True
                    if k==0:
                        warm_start=False
                    MILP_data_point=self.get_MILP_data(b=param,warm_start=warm_start)
                    if bool(MILP_data_point):
                        if self.save:
                            MILP_data_list=[MILP_data_point.b, MILP_data_point.x_opt, MILP_data_point.y_opt, tuple(MILP_data_point.cov_set)]
                            MILP_LPs=[[LP.b, LP.lb, LP.ub, LP.x, LP.y, LP.lmbd, LP.kappa, LP.tau, LP.cost, LP.feas] for LP in MILP_data_point.cover]
                            MILP_data_list.append(MILP_LPs)
                            pickled_1=pkl.dumps(MILP_data_list)
                            optimized_pickle_1=pickletools.optimize(pickled_1)
                            f1.write(optimized_pickle_1)
                            f1.flush()
                            
                            pickled=pkl.dumps([param["x0"],MILP_data_point.opt_sp.y,MILP_data_point.cov_set])
                            optimized_pickle=pickletools.optimize(pickled)
                            f.write(optimized_pickle)
                            f.flush()
                        self.add_MILP_problem(MILP_data_point)
                        u_opt=np.array([MILP_data_point.opt_sp.x[self.P.A.shape[0]*(self.P.N+1)],MILP_data_point.opt_sp.x[self.P.A.shape[0]*(self.P.N+1)+1]])
                        x0_k=np.dot(self.P.A,np.array(param['x0']).reshape(self.P.nx,1))+np.dot(self.P.B,u_opt)
                        param["x0"]=np.array(np.transpose(x0_k)).reshape(self.P.nx,).tolist()  

                while not self.check_prob_bnd():            
                    param=self.sample_MILP_parameter()  #   Succesive Problems
                    for k in range(self.N_prop):
                        warm_start=True
                        if k==0:
                            warm_start=False
                        MILP_data_point=self.get_MILP_data(param,warm_start=warm_start)
                    
                        if bool(MILP_data_point):
                            if self.save:
                                MILP_data_list=[MILP_data_point.b, MILP_data_point.x_opt, MILP_data_point.y_opt, MILP_data_point.cov_set]
                                MILP_LPs=[[LP.b, LP.lb, LP.ub, LP.x, LP.y, LP.lmbd, LP.kappa, LP.tau, LP.cost, LP.feas] for LP in MILP_data_point.cover]
                                MILP_data_list.append(MILP_LPs)
                                pickled_1=pkl.dumps(MILP_data_list)
                                optimized_pickle_1=pickletools.optimize(pickled_1)
                                f1.write(optimized_pickle_1)
                                f1.flush()
                                pickled=pkl.dumps([param["x0"],MILP_data_point.opt_sp.y,MILP_data_point.cov_set])
                                optimized_pickle=pickletools.optimize(pickled)
                                f.write(optimized_pickle)
                                f.flush()
                            self.add_MILP_problem(MILP_data_point)
                            u_opt=np.array([MILP_data_point.opt_sp.x[self.P.A.shape[0]*(self.P.N+1)],MILP_data_point.opt_sp.x[self.P.A.shape[0]*(self.P.N+1)+1]])
                            x0_k=np.dot(self.P.A,np.array(param['x0']).reshape(self.P.nx,1))+np.dot(self.P.B,u_opt)
                            param["x0"]=np.array(np.transpose(x0_k)).reshape(self.P.nx,).tolist()
                        else:
                            break


    def get_MILP_data(self, b,warm_start=False):    
        '''
        Function that returns an MILP data point for parameter b
        '''
        if len(self.MILP_covers)>self.N_prop:
            check_covs=True
        else:
            check_covs=False 
        MILP_data_point=self.P.getMILP_Data(b, self.MILP_list, len(self.MILP_covers), check_covs,warm_start)
        return MILP_data_point    
                                

    def add_MILP_problem(self, MILP_data,load_data=False):
        '''
         Function that manage the adding of the MILP data point 
         and update the Good-Turing Estimator G
        '''                                                    
        if load_data or(MILP_data not in self.MILP_list): 
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
        if not load_data:
            print("G:"+str(np.round(self.G,3))+" N:"+str(self.N)+" N1:"+str(self.N1)+ \
                  " len: "+str(len(self.MILP_covers[-1]))+" state: "+str(MILP_data.b[0:4].T.squeeze()))
    

    def check_prob_bnd(self):
        if self.G+self.c*np.sqrt(1/self.N*np.log(3/self.beta))<=self.eps:
            return True
        else:
            return False
    

    def sample_MILP_parameter(self):
        b=self.P.sample_params(self.rng)
        return b #dictionary
    

    def load_dataset(self,dataset_path):
        '''
        Function that allows the loading of previous created dataset
        '''
        with gzip.open(dataset_path, "rb") as f1:
            while True:
                try:
                    p=pkl.Unpickler(f1)
                    MILP_data_list=p.load()
                    b=MILP_data_list[0]
                    x_opt=MILP_data_list[1]
                    y_opt=MILP_data_list[2]
                    MILP_cover=[LP_SP_data(b=k[0],lb=k[1],ub=k[2],x=k[3],y=k[4],\
                                           lmbd=k[5],kappa=k[6],tau=k[7],cost=k[8],feas=k[9]) for k in MILP_data_list[-1]]
                    for sp in MILP_cover:
                        if(sp.x==x_opt).all() and(sp.y==y_opt).all():
                            opt_sp=sp
                            break
                    MILP_data_point=MILP_data(b=b,opt_sp=opt_sp,cover=MILP_cover)
                    self.add_MILP_problem(MILP_data_point,load_data=True)
                except EOFError:
                    break



class base_prob:
    '''
    Root Class for Generic MILP solution data collection
    '''

    def __init__(self):
        self.model=[]
        self.vars=vars()
        self.opt_vars=[]
        self.opt_vars_sp=[]
        self.LP_SP=LP_SP_Builder()  #   Generic LP subproblem 
        
            
    def solve_(self):
        
        return{}
    
    def addVariable(self,dimensions,var_type,var_string,solve_type="full" ,lb=None, ub=None,LP_sol_bin=None):
        '''
        Method for adding single multidimensional Variable to the model  
        defined with SCIP or CVXOPT depending on the solve_type
        with dimensions= [size of variable per stage, number of stages, number of variables per stage]
        '''

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
                                    if (lb_bnd==ub_bnd): 
                                        m.addCons(lb_bnd<=x[i,j,k], name="c_lb"+var_name)
                                        m.addCons(ub_bnd>=x[i,j,k],name="c_ub"+var_name)
                                    else:
                                        if LP_sol_bin[var_name]==1.0:
                                            m.addCons(x[i,j,k]>=lb_bnd,name="c_lb"+var_name)
                                            m.addCons(x[i,j,k]<=LP_sol_bin[var_name],name="c_lb="+var_name)
                                            m.addCons(x[i,j,k]>=ub_bnd,name="c_ub"+var_name)
                                        else:
                                            m.addCons(lb_bnd>=x[i,j,k],name="c_lb"+var_name)
                                            m.addCons(LP_sol_bin[var_name]<=x[i,j,k],name="c_ub="+var_name)
                                            m.addCons(x[i,j,k]<=ub_bnd,name="c_ub"+var_name)

                        else:                             
                            x[i,j,k] = m.addVar( vtype=var_type,name=var_string+"(%s,%s,%s)"%(i,j,k),lb=lb,ub=ub)
        else:
            raise Exception("More than 3 dimensions not supported currently")
        
        if solve_type=="full":
            self.opt_vars.append(self.vars[var_string])
        elif solve_type=="sp":
            self.opt_vars_sp.append(self.vars[var_string])

        

class OA_MIMPC(base_prob):
    '''
    Class for Obstacle Avoidance Problem
    '''

    def __init__(self,Q = 1000,
                R = 50, N_obs_max=10, N=40, 
                lb_x=np.array([-3,-3,-2,-2]),
                ub_x=-np.array([-3,-3,-2,-2]),
                lb_u=np.array([-1,-1]),
                ub_u=-np.array([-1,-1]),dt=0.1):

        super(OA_MIMPC,self).__init__()
        self.N_obs_max=N_obs_max    #   Maximum number of Obstacles
        self.lb_x=lb_x              #   State and input constraints
        self.lb_u=lb_u
        self.ub_x=ub_x
        self.ub_u=ub_u
        self.dt=dt                  #   Horizon and sampling time
        self.N=N
        #   Dynamic of the Robot
        self.A = np.matrix([[1, 0,dt, 0],[0, 1, 0, dt],[0, 0, 1, 0],[0, 0, 0, 1]])  
        self.B = dt*np.matrix([[0, 0],[0, 0],[1, 0],[0, 1]])
        self.nx = self.A.shape[1]   #    Number of states
        self.nu = self.B.shape[1]   #    Number of inputs
        self.free_x=[]              #   Vectors for collecting the free response
        self.free_u=np.zeros((self.nu,self.N))
        self.free_cost=0
        self.Q=Q*np.identity(self.nx)   #   Cost Matrices
        self.R=R*np.identity(self.nu)
        self.space_dim=2                #   State pace dimension



    def getMILP_Data(self,parameters, MILP_list,check_old_cov,warm_start): 
        
        '''
         This method return an MILP data structure containing the strategy s(b) for the problem at hand 
        '''
        x0_p=np.array(parameters["x0"])      #Problem parameters
        ol_p=parameters["ol"]
        obs_size_p=parameters["obs_size"]
        n_obs=ol_p.shape[1]
        prev_bin_sol=None                   #Previous binary solution for warm start
        if n_obs<self.N_obs_max:
            ol_p=np.concatenate((ol_p, self.ub_x[0]*np.ones((2,self.N_obs_max-n_obs))),axis=1)
            obs_size_p=np.concatenate((obs_size_p, np.zeros((2,self.N_obs_max-n_obs))),axis=1)
        
        if(bool(MILP_list)):                
            prev_bin_sol=MILP_list[-1].y_opt
        
        # Problem solution

        P_sol=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,warm_start=warm_start,prev_bin_sol=prev_bin_sol)
        b=np.vstack((x0_p.reshape((-1,1)),ol_p.reshape((-1,1)),obs_size_p[0].reshape((-1,1))))
        n_hndlr=nodelist_Handler(self.nodelist)
        self.n_hndlr=n_hndlr
        costs=[]
        for l in n_hndlr.leaves:
            costs.append(l["objval"])
        

        LP_SP_data_list=[]
        differences=[]
        solution_found=False
        i=0
        min_diff_node=None
        
        if P_sol["status"]=="optimal" :     #    If SCIP doesn't return the optimal node we create it 
                                            #    solving the LP with binary variables fixed to the full binary solution 

            bin_sol={key:round(P_sol["binary_sol"][key]) for key in P_sol["binary_sol"].keys()}
            for l in n_hndlr.leaves:
            
                if abs(l["objval"]-P_sol["optimal_cost"])<1e-4:
                    
                    diff=0.0
                    for var in P_sol["binary_sol"].keys():
                        diff=max([diff, abs(l['LPsol_bin'][var]-P_sol["binary_sol"][var])])
                    differences.append(diff)
                    if differences[i]==min(differences):
                        min_diff_node=l
                    i=i+1
                
            if not bool(min_diff_node): 
                LP_lb={}
                LP_ub={}

                for y in n_hndlr.leaves[0]["ub"].keys(): 
                
                    LP_lb[y]=bin_sol[y]
                    LP_ub[y]=bin_sol[y]
       
                P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp",LP_sol_bin=bin_sol) 
                objval=P_sol_lp["optimal_cost"]
                dualval=objval
                binsol=P_sol_lp["binary_sol"]
                xsol=P_sol_lp["x_sol"]
                usol=P_sol_lp["u_sol"]
                nodedict = {                
                            "number": -1,
                            "type":0,
                            "leavesNodesNumber":0,
                            "LPsol_bin": binsol,
                            "x_sol":xsol,
                            "u_sol":usol,
                            "LPdual_sol":P_sol_lp["lambda_sol"],
                            "objval": objval,
                            "parent": 1,
                            "node":None,
                            "age": n_hndlr.leaves[0]["age"],
                            "depth": n_hndlr.leaves[0]["depth"],
                            "lb":LP_lb,
                            "ub":LP_ub,
                            "constraints": n_hndlr.leaves[0]["constraints"],
                            "primalbound": objval,
                            "dualbound": dualval}
                n_hndlr.nodelist.append(nodedict)
                n_hndlr.leaves.append(nodedict)
                n_hndlr.nodesId=n_hndlr.nodesId|set([nodedict["number"]])
                solution_found=True
                n_hndlr.feasibleLeavesId=n_hndlr.feasibleLeavesId|set([nodedict["number"]])
                min_diff_node=n_hndlr.leaves[-1]
                solution_node=min_diff_node

            else:
                min_diff_node['LPsol_bin']=P_sol["binary_sol"]
                solution_node=min_diff_node
                solution_found=True
        else:

            return None
    
        n_hndlr.recursive_make_cover(ref_sp=solution_node)      #   Cover completion 

        #   Previous Cover Reuse

        sol_lb=solution_node["lb"]; sol_ub=solution_node["ub"]  
        binary_vars=list(sol_lb.keys())
        sol_sp=tuple([tuple(list(sol_lb.values())), tuple(list(sol_ub.values()))])
        new_cov=None
        MILP_covers_bins=[[MC.cov_set, MC.y_opt] for MC in MILP_list if np.linalg.norm(MC.b[0:self.space_dim].T-x0_p[0:self.space_dim])<=2]
        b_sol_array=np.array(list(bin_sol.values()))
        b_sol_tuple=tuple(list(bin_sol.values()))
        for cov_bin in MILP_covers_bins:
            lb_ub=[k for t in cov_bin[0] for k in t]    #   Vector with all lb ub in the problem at hand
            lb_ub_intrvl=[[np.array(list(t[0])),np.array(list(t[1]))] for t in cov_bin[0]]
            #   (binary soln. observed before) or     (Optimal subproblem in old cover)    or (Binary soln. on the margin of old cover)
            if (b_sol_array==cov_bin[1]).all() or (sol_sp in cov_bin[0] and check_old_cov) or b_sol_tuple in lb_ub:
                sol_lbub=[lb_ub for lb_ub in lb_ub_intrvl if (b_sol_array>=lb_ub[0]).all() and (b_sol_array<=lb_ub[1]).all()]
                sol_lbub=sol_lbub[0]
                sol_lb={k:float(sol_lbub[0][i]) for i,k in  enumerate(list(solution_node["lb"].keys()))}
                sol_ub={k:float(sol_lbub[1][i]) for i,k in  enumerate(list(solution_node["ub"].keys()))}
                P_sol_lp_fix=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=sol_lb, ub=sol_ub, solve_type="sp", LP_sol_bin=bin_sol)
                P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=sol_lb, ub=sol_ub, solve_type="sp")
                if P_sol_lp_fix["optimal_cost"]-P_sol_lp["optimal_cost"]<=1e-2:
                    new_cov=cov_bin
                    break
        #   Solving all the problems in the previous cover for getting LPs information for current problem
        if new_cov is not None and solution_found:  
            for sp in new_cov[0]:
                LP_lb={}
                LP_ub={}
                lb=sp[0]
                ub=sp[1]
                for i in range(len(binary_vars)):
                    LP_lb.update({binary_vars[i]:lb[i]})
                    LP_ub.update({binary_vars[i]:ub[i]})
                if (b_sol_array>=sp[0]).all() and (b_sol_array<=sp[1]).all():
                    P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp", LP_sol_bin=bin_sol)                  
                else:
                    P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp")    

                if P_sol_lp["status"]!="infeasible":
                    LP_SP_data_point=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                        x=np.vstack((P_sol_lp["x_sol"].reshape((-1,1)),P_sol_lp["u_sol"].reshape((-1,1)))),
                        y=np.array(list(P_sol_lp["binary_sol"].values())),lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),
                        kappa=0.0, tau=1.0,cost=P_sol_lp["optimal_cost"],feas=P_sol_lp["status"])
                    if(b_sol_array>=sp[0]).all() and (b_sol_array<=sp[1]).all():
                        opt_sp=LP_SP_data_point   
                else:
                    # If the problem is infeasible we attach default LP_SP_data correponding to free evoulution 
                    objval=self.free_cost   
                    dualcost=1.0
                    LP_SP_data_point=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                                                x=np.vstack((self.free_x.reshape((-1,1)),self.free_u.reshape((-1,1)))), 
                                                y=np.zeros((len(LP_lb.values()),1)),
                                                lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),
                                                kappa=objval-dualcost, tau=0.0,
                                                cost=objval,feas=P_sol_lp["status"])
                LP_SP_data_list.append(LP_SP_data_point)
        #  Solving all the problems in the current cover for getting LPs information       
        elif new_cov is None:   
            for l in n_hndlr.leaves: 
                LP_lb=l["lb"]
                LP_ub=l["ub"]        
                if(l["number"] in n_hndlr.feasibleLeavesId): #  If the leaf is feasible we solve fixing the binary variables 
                    LP_sol_bin=l["LPsol_bin"]                #  to the leaf solution for binaries to get LP informations

                    P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp", LP_sol_bin=LP_sol_bin)
                    try:
                        LP_SP_data_point=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                            x=np.vstack((P_sol_lp["x_sol"].reshape((-1,1)),P_sol_lp["u_sol"].reshape((-1,1)))),
                            y=np.array(list(P_sol_lp["binary_sol"].values())),lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),
                            kappa=0.0, tau=1.0,cost=P_sol_lp["optimal_cost"],feas=P_sol_lp["status"])
                    except:
                        print('warning')

                    if(l==solution_node):
                        opt_sp=LP_SP_data_point                   
                else:   
                    # Getting infeasibility certificate and resolving for added problems  
                    P_sol_lp=self.solve_(x0=x0_p,ol=ol_p,obs_size=obs_size_p,lb=LP_lb, ub=LP_ub, solve_type="sp")
                    if P_sol_lp['status']!='infeasible':
                        LP_SP_data_point=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                        x=np.vstack((P_sol_lp["x_sol"].reshape((-1,1)),P_sol_lp["u_sol"].reshape((-1,1)))),
                        y=np.array(list(P_sol_lp["binary_sol"].values())),lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),
                        kappa=0.0, tau=1.0,cost=P_sol_lp["optimal_cost"],feas=P_sol_lp["status"])
                    else:
                        objval=self.free_cost
                        dualcost=1.0
                        LP_SP_data_point=LP_SP_data(b=b,lb=LP_lb,ub=LP_ub,
                            x=np.vstack((self.free_x.reshape((-1,1)),self.free_u.reshape((-1,1)))),
                            y=np.zeros((len(LP_lb.values()),1)), lmbd=P_sol_lp["lambda_sol"].reshape(-1,1),
                            kappa=objval-dualcost, tau=0.0,cost=objval,feas=P_sol_lp['status'])
                LP_SP_data_list.append(LP_SP_data_point)
        try:
            opt_sp=opt_sp
        except:
            with open("errorprobsp.p", "wb") as f:
                pkl.dump(parameters, f)
            print("warning")

        return MILP_data(b=b,opt_sp=opt_sp,cover=LP_SP_data_list)
          

    def solve_(self, x0=np.array([10, 6.5,0,0]), 
                    ol=np.array([[1,1,3,9,10],[3,-1,2,5,5]]),
                    obs_size=np.array([[1,1,1,1,2],[2,2,2,2,1]]),
                    solve_type='full',
                    lb=None, ub=None,LP_sol_bin=None,warm_start=False,prev_bin_sol=None):
        '''
        This method setup and solve an LP or MILP for OA based on solve_type 
        and return the correpondent dictionary with solution information
        '''
            
        self.M=2*self.ub_x[0]
        self.add_variables(solve_type=solve_type,lb=lb, ub=ub, LP_sol_bin=LP_sol_bin)   #   Problem setup
        if(warm_start):
            self.warm_start_binary(prev_bin_sol=prev_bin_sol)   # Using previous binary solution 
        self.add_system_constraints(x0,solve_type=solve_type)
        self.add_OA_constraints(ol=ol,obs_size=obs_size,solve_type=solve_type)
        self.add_cost(solve_type=solve_type) 
    
        if solve_type=="full":                  #   Problem solution for an MILP
     
            m=self.model
            self.nodelist=[]
            eventhdlr = LPstatEventhdlr()       #   Event Handler for node information
            eventhdlr.nodelist = self.nodelist  #   List with all node information            
                  
            m.disablePropagation()              #   SCIP Parameters setting for pure BnB   
            m.setSeparating(3)
            m.setPresolve(3)
            m.setHeuristics(3)
            m.setParam("limits/time","120")
            m.setParam("misc/usesymmetry","0")
            m.setParam("conflict/enable","0")
            m.setParam("lp/disablecutoff","1")
            m.setParam("propagating/maxrounds","0")
            m.setParam("propagating/maxroundsroot","0")
            m.setParam("heuristics/clique/freq","10")
            m.setParam("heuristics/conflictdiving/freq","10")
            m.setParam("heuristics/farkasdiving/freq","10")
            m.setParam("heuristics/rounding/freq","1")
            m.setParam("numerics/feastol","1e-5")
    
            m.includeEventhdlr(eventhdlr, "LPstat", 
                               "generate LP statistics after every LP event")  # Attach event handler for node info
            m.hideOutput()
            m.optimize()
            P_sol=self.retreive_sol()

        elif solve_type=="sp":                  #   Problem solution for an LP

            self.LP_SP.solve__()                  
            P_sol=self.retreive_sol_cvxopt()
        return P_sol    
    

    def warm_start_binary(self,prev_bin_sol):
        
        m=self.model
        tx,tu,x,u,bl,bu=self.opt_vars
        sol = m.createSol()
        theta=0
        for i in range(self.space_dim):
                for j in range(self.N):
                    for k in range(self.N_obs_max):
                        m.setSolVal(sol, bl[i,j,k], prev_bin_sol[theta])
                        m.setSolVal(sol, bu[i,j,k], prev_bin_sol[theta+self.space_dim*self.N*self.N_obs_max])
                        theta+=1
        m.addSol(sol)
                  

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
        self.addVariable([self.space_dim, self.N, self.N_obs_max],lb=lb, ub=ub, var_type='B',var_string='bl',\
                         solve_type=solve_type,LP_sol_bin=LP_sol_bin)
        self.addVariable([self.space_dim, self.N, self.N_obs_max],lb=lb, ub=ub, var_type='B',var_string='bu',\
                         solve_type=solve_type,LP_sol_bin=LP_sol_bin)
   
                  
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
            for k in range(self.N):
                sum_x=[abs(cp.sum([(1+100*(k==self.N-1))*float(self.Q[i,j])*x[j,k+1] for j in range(self.nx)])) \
                       for i in range(self.nx)]
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
                                            +cp.sum([float(self.B[i,j])*u[j,k-1] for j in range(self.nu)]),\
                                                name="c_dyn_(%s,%s)"%(i,k-1))                        
                          
                    m.addCons(x[i,k]<=float(self.ub_x[i]), name="c_xub_(%s,%s)"%(i,k))
                    m.addCons(x[i,k]>=float(self.lb_x[i]), name="c_xlb_(%s,%s)"%(i,k))
                    if i<self.nu:   #   Assuming nu<nx
                        m.addCons(u[i,k-1]<=float(self.ub_u[i]), name="c_uub_(%s,%s)"%(i,k-1))
                        m.addCons(u[i,k-1]>=float(self.lb_u[i]), name="c_ulb_(%s,%s)"%(i,k-1))  


    def add_OA_constraints(self, ol=np.array([[1,1,3,9,10],[3,-1,2,5,5]]),obs_size=np.array([[1,1,1,1,2],[2,2,2,2,1]])
                           ,solve_type="full"):

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
        '''
        Method for building a dictionary with solution information from SCIP
        '''

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
        '''
        Method for building a dictionary with solution information from CVXOPT
        '''

        m=self.LP_SP
        x,u,bl,bu=self.opt_vars_sp
        return_dict={}
        binary_sol={}
        LPsol_full={}
        Variables=m.prob.variables()
        lambda_vals=[float(m.constraint_dict[c].multiplier.value[0]) for c in m.constraint_dict.keys() if "=" not in c]
        return_dict.update({"lambda_sol":np.array(lambda_vals)})
        objval=m.prob.objective.value()                     
        if(objval==None): 
            status="infeasible"
            return_dict.update({"status":status})      
            return_dict.update({"optimal_cost":objval})
            return_dict.update({"x_sol":self.free_x.reshape((-1,1))})
            return_dict.update({"u_sol":self.free_u.reshape((-1,1))})
            for i in range(self.space_dim):
                for j in range(self.N):
                    for k in range(self.N_obs_max):
                        binary_sol[bl[i,j,k].name]=float(0)
                        
            for i in range(self.space_dim):
                for j in range(self.N):
                    for k in range(self.N_obs_max):
                        binary_sol[bu[i,j,k].name]=float(0)                        

            return_dict.update({"binary_sol":binary_sol})
            return_dict.update({"optimal_cost":self.free_cost})            
        else:
            status="optimal"
            return_dict.update({"status":status})
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
                        if((bl[i,j,k].value[0]<=1e-4 and bl[i,j,k].value[0]>=-1e-4) or \
                           (bl[i,j,k].value[0]<=1+1e-4 and bl[i,j,k].value[0]>=1-1e-4)):
                            binary_sol[bl[i,j,k].name]=round(float(bl[i,j,k].value[0]))
                        else:    
                            binary_sol[bl[i,j,k].name]=float(bl[i,j,k].value[0])
                        
            for i in range(self.space_dim):
                for j in range(self.N):
                    for k in range(self.N_obs_max):
                        if((bu[i,j,k].value[0]<=1e-4 and bu[i,j,k].value[0]>=-1e-4) or \
                           (bu[i,j,k].value[0]<=1+1e-4 and bu[i,j,k].value[0]>=1-1e-4)):
                            binary_sol[bu[i,j,k].name]=round(float(bu[i,j,k].value[0]))
                        else:    
                            binary_sol[bu[i,j,k].name]=float(bu[i,j,k].value[0])
            return_dict.update({"binary_sol":binary_sol})       
            for var in Variables:
                LPsol_full[var.name]=float(var.value[0])
            return_dict.update({"LPsol_full":LPsol_full})

        return return_dict
    

    def sample_params(self, rng):
        '''
        Function for random parameter generation
        '''

        def generate_initial_condition(ol_list,ou_list,ub_x,lb_x,rng):
            ul_p=ub_x[0]
            ll_p=lb_x[0]
            x0=rng.uniform(ll_p,ul_p,2)
            def check_in_obs(x):
                for i in range(ol_list.shape[1]):
                    if (x[0]>=ol_list[0][i] and x[1]>=ol_list[1][i]) and (x[0]<=ou_list[0][i] and x[1]<=ou_list[1][i]):
                        return True
                return False     
            while check_in_obs(x0):
                x0=rng.uniform(ll_p,ul_p,2)
            return list(x0)+[0.0,0.0]
        
        ol_list=np.array([[-2.5,-2.,1, 2 ],[-2.5 ,1,-2, 1.5 ]])
        obs_size_list=np.array([[2.5,2,2, 1],[1.5,2,2.5, 1.5]])
        ou_list=ol_list+obs_size_list
        x0=generate_initial_condition(ol_list,ou_list,self.ub_x,self.lb_x,rng)
        b={"x0":x0, "ol": ol_list, "obs_size": obs_size_list}

        return b



if __name__=="__main__":
   
    N_obs_max=4
    N=20
    dt=0.1
    P=OA_MIMPC(N=40, dt=0.1, N_obs_max=N_obs_max)
    Dataset_creator=MILP_DataSet(P=P, save_data=True)
    Dataset_creator.make_Dataset()
   
            


                



        

    


        


    

    
