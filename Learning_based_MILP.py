from logging import NullHandler
from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, quicksum
import numpy as np
import math
from dataclasses import dataclass, field

@dataclass
class LP_SP_data:
    b    : np.ndarray
    lb   : np.ndarray
    ub   : np.ndarray

    x    : np.ndarray
    y    : np.ndarray
    lmbd : np.ndarray

    cost : np.ndarray

    feas : bool






@dataclass
class MILP_data:
    b      : np.ndarray

    opt_sp : LP_SP_data
    cover  : List[LP_SP_data]

    cov_set: List[nd.ndarray]   = field(init=False, repr=False)

    x_opt  : np.ndarray         = field(init=False, repr=False) 
    y_opt  : np.ndarray         = field(init=False, repr=False)

    feas_nodes: List[LP_SP_data]= field(init=False, repr=False)


    def __post_init__(self):

        x_opt=opt_sp.x
        y_opt=opt_sp.y

        cov_set=[]
        feas_nodes=[]
        for sp in cover:
            cov_set.append(np.array([sp.lb, sp.ub]))
            if sp.feas:
                feas_nodes.append(sp)





#EventHandler for collecting node information

class LPstatEventhdlr(Eventhdlr):
    """PySCIPOpt Event handler to collect data on LP events."""

    vars = {}
    # nodelist=[]
    def collectNodeInfo(self):

        objval = self.model.getSolObjVal(None)

         # Collecting Solutions for all the variabeles and the binary variables for current Node
        LPsol_bin,LPsol_full = {},{}
        
        if self.vars == {}:
            self.vars = self.model.getVars(transformed=False)
        
       
        for var in self.vars:
            solval = self.model.getSolVal(None, var)
            LPsol_full[var.name]=self.model.getSolVal(None, var)
            if(var.vtype())=='BINARY':
                LPsol_bin[var.name] = self.model.getSolVal(None, var)

        node = self.model.getCurrentNode()
        vars=self.model.getVars()
        lb={}
        ub={}
        
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
            parent = 1

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
            "LPsol_full":LPsol_full,
            "LPdual_sol":LPdual_sol,
            "objval": objval,
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

    def eventexec(self, event):
        self.collectNodeInfo()    
        return {}
    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

class base_prob:

    def __init__(self,**kwargs):
        self.model=Model()
        self.vars=vars()
        self.opt_vars=[]
        self.opt_vars_sp=[]
        self.LP_sp_model=Model()
        return{}
    
    def solve(self):
        
        return{}
    def addVariable(self,dimensions,var_type,var_string, lb=None, ub=None, solve_type="full"):
        # dimensions= [size of variable per stage, number of stages, number of variables per stage]
        n_dim=len(dimensions)
        self.vars[var_string]={}
        x=self.vars[var_string]
        if solve_type=="full":
            m=self.model
        elif solve_type=="sp":
            m=self.LP_sp_model
        if n_dim==1:
            for i in range(dimensions[0]):
                x[i] = m.addVar( vtype=var_type,name=var_string+"(%s)"%(i),lb=lb,ub=ub)
        elif n_dim==2:
            for i in range(dimensions[0]):
                for j in range(dimensions[1]):
                    x[i,j] = m.addVar( vtype=var_type,name=var_string+"(%s,%s)"%(i,j),lb=lb,ub=ub)
        elif n_dim==3:
            for i in range(dimensions[0]):
                for j in range(dimensions[1]):
                    for k in range(dimensions[2]):
                        var_name=var_string+"(%s,%s,%s)"%(i,j,k)
                        if solve_type=="sp":
                            try:
                                lb_var=lb[var_name][1]
                                ub_var=ub[var_name][1]
                                x[i,j,k] = m.addVar( vtype=var_type,name=var_string+"(%s,%s,%s)"%(i,j,k),lb=lb_var,ub=ub_var)
                            except:
                                raise Exception("Bounds only supported for binary variables")
                        else:
                            x[i,j,k] = m.addVar( vtype=var_type,name=var_string+"(%s,%s,%s)"%(i,j,k),lb=lb,ub=ub)

        else:
            raise Exception("More than 3 dimensions not supported currently")

        if solve_type=="full":
            self.opt_vars.append(self.vars[var_string])
        elif solve_type=="sp":
            self.opt_vars_sp.append(self.vars[var_string])
        



    def getCoverSetsBounds(self, leaves):
        ## Step 1) Check if cover using is_cover on leaves
        ## Step 2) Make cover if not, and obtain solution of modified lp_sp if necessary
        ## Step 3)  Update leaves, feasible, infeasible
        return{}
    
    def add_data_point(self, data):
        return 


class OA_MIMPC(base_prob):
    def __init__(self, Q,R, N_obs_max=10, N=30, 
                lb_x=np.array([-15,-15,-8,-8]),
                ub_x=-np.array([-15,-15,-8,-8]),
                lb_u=np.array([-5,-5]),
                ub_u=-np.array([-5,-5]) ):

        self.Q=Q
        self.R=R
        self.N_obs_max=N_obs_max
        self.lb_x=lb_x
        self.lb_u=lb_u
        self.ub_x=ub_x
        self.ub_u=ub_u
        self.N=30
        self.A = np.matrix([[1, 0, 0.1, 0],[0, 1, 0, 0.1],[0, 0, 1, 0],[0, 0, 0, 1]])
        self.B = 0.1*np.matrix([[0, 0],[0, 0],[1, 0],[0, 1]])

        self.nx = self.A.shape[1]#% Number of states
        self.nu = self.B.shape[1]#% Number of inputs

        self.space_dim=2
        


        


    def add_variables(self, solve_type='full', lb=None, ub=None):
        if solve_type=="full":
            self.addVariable([self.nx, self.N+1], 'C','x')
            self.addVariable([self.nu, self.N], 'C','u')
            self.addVariable([self.N], 'C','tx')
            self.addVariable([self.N], 'C','tu')
            self.addVariable([self.space_dim, self.N, self.N_obs_max], 'B','bl')
            self.addVariable([self.space_dim, self.N, self.N_obs_max], 'B','bu')
        elif solve_type=="sp":
            self.addVariable([self.nx, self.N+1], 'C','x', solve_type=solve_type)
            self.addVariable([self.nu, self.N], 'C','u', solve_type=solve_type)
            self.addVariable([self.N], 'C','tx', solve_type=solve_type)
            self.addVariable([self.N], 'C','tu', solve_type=solve_type)
            self.addVariable([self.space_dim, self.N, self.N_obs_max], 'C','bl', lb, ub, solve_type=solve_type)
            self.addVariable([self.space_dim, self.N, self.N_obs_max], 'C','bu', lb, ub, solve_type=solve_type)


    def add_cost(self, solve_type="full"):
        
        cost=0
        for k in range(self.N):
            cost+=tx[k]+tu[k]
        if solve_type=="full":
            m=self.model
            x,u,tx,tu,bl,bu=self.opt_vars
        elif solve_type=="sp":
            m=self.LP_sp_model
            x,u,tx,tu,bl,bu=self.opt_vars_sp
        m.setObjective(cost,"minimize")

    def add_system_constraints(self, x0, solve_type="full"):
        self.x0=x0
        if solve_type=="full":
            m=self.model
            x,u,tx,tu,bl,bu=self.opt_vars
        elif solve_type=="sp":
            m=self.LP_sp_model
            x,u,tx,tu,bl,bu=self.opt_vars_sp
        for k in range(self.N+1):
            for i in range(self.nx):
                if k==0:
                    m.addCons(x[i,k]==x0[i])
                else:
                    
                    m.addCons((1+100*(k==self.N))*self.Q[i,i]*x[i,k]<=tx[k-1])
                    m.addCons(-(1+100*(k==self.N))*self.Q[i,i]*x[i,k]<=tx[k-1])
                    m.addCons(x[i,k]==quicksum(self.A[i,j]*x[j,k-1] for j in range(self.nx))\
                                      +quicksum(self.B[i,j]*u[j,k-1] for j in range(self.nu)))
                    m.addCons(x[i,k]<=self.ub_x[i])
                    m.addCons(x[i,k]>=self.lb_x[i])
                    if i<self.nu:#assuming nu<nx
                        m.addCons(self.R[i,i]*u[i,k-1]<=tu[k-1])
                        m.addCons(-self.R[i,i]*u[i,k-1]<=tu[k-1])
                        m.addCons(u[i,k-1]<=self.ub_u[i])
                        m.addCons(u[i,k-1]>=self.lb_u[i])

    def add_OA_constraints(self, ol=np.array([[1,1,3,9,10],[3,-1,2,5,5]]),obs_size=np.array([[1,1,1,1,2],[2,2,2,2,1]]), solve_type="full"):
                    
        n_obs=ol.shape[1]
        M=self.M
        if n_obs<self.N_obs_max:
            self.ol=np.concatenate((ol, M*np.ones((2,self.N_obs_max-n_obs))),axis=1)
            self.obs_size=np.concatenate((obs_size, np.zeros((2,self.N_obs_max-n_obs))),axis=1)
    
        ou=ol+obs_size

  
        if solve_type=="full":
            m=self.model
            x,u,tx,tu,bl,bu=self.opt_vars
        elif solve_type=="sp":
            m=self.LP_sp_model
            x,u,tx,tu,bl,bu=self.opt_vars_lp

        for k in range(self.N):
            for i in range(self.space_dim):
                for l in range(self.N_obs_max):
                    m.addCons(x[i,k+1]<=ol[i,l]+M*bl[i,k,l])
                    m.addCons(x[i,k+1]>=ou[i,l]-M*bu[i,k,l])
                    m.addCons(bl[0,k,l]+bl[1,k,l]+bu[0,k,l]+bu[1,k,l]<=2*2-1)

    def solve(self, x0, 
                    ol=np.array([[1,1,3,9,10],[3,-1,2,5,5]]),
                    obs_size=np.array([[1,1,1,1,2],[2,2,2,2,1]]),
                    solve_type='full',
                    lb=None, ub=None):
        # solve_type=='full' or 'sp'
        self.M=np.max(np.abs(ol))+np.max(np.abs(obs_size))+5
        self.add_variables(solve_type, lb, ub)
        self.add_cost(solve_type)
        self.add_system_constraints(x0,solve_type)
        self.add_OA_constraints(ol,obs_size,solve_type)
        if solve_type=="full":
            m=self.model
            self.nodelist=[]
            eventhdlr = LPstatEventhdlr()
            eventhdlr.nodelist = self.nodelist
            # model.readProblem('/home/luirusso/LearnWSL/model.cip')
            m.disablePropagation()
            m.setSeparating(3)
            m.setPresolve(3)
            m.setHeuristics(3)
            m.setParam("lp/disablecutoff","1")
            m.includeEventhdlr(
            eventhdlr, "LPstat", "generate LP statistics after every LP event"
                    )
        elif solve_type=="sp":
            m=self.LP_sp_model 

        m.optimize()


    def retreive_sol(self, solve_type="full"):
        if solve_type=="full":
            m=self.model
            x,u,tx,tu,bl,bu=self.opt_vars
        elif solve_type=="sp":
            m=self.LP_sp_model
            x,u,tx,tu,bl,bu=self.opt_vars_lp

        return_list=[]
        Variables=m.getVars(transformed=False)
        binary_sol={}
        b_l_sol, b_u_sol=[],[]
        x_sol=[self.x0]
        u_sol=[]
        status=m.getStatus
        if(status=="infeasible"): 
            status="infeasible"
            return_list.append(status)           
        else:
            status="optimal"
            return_list.append(status)
            for i in range(self.N):
                x_sol.append([])
                u_sol.append([])
                for j in range(self.nx):
                    x_sol[i+1].append(m.getVal(x[j,i+1]))
                for j in range (self.nu):
                    u_sol[i].append(m.getVal(u[j,i]))
            return_list.append(np.array(x_sol))
            return_list.append(np.array(u_sol))
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
                        binary_sol[bu[i,j,k].name]=m.getVal(bu[i,j,k])
            return_list.append(b_l_sol)
            return_list.append(b_u_sol)
            return_list.append(binary_sol)
            if solve_type=="sp":
                LPsol_full={}
                for var in Variables:
                    solval = m.getVal(var)
                    # store only solution values above 1e-6
                    # # if abs(solval) > 1e-6:
                    LPsol_full[var.name]=m.getVal(var)
                return_list.append(LPsol_full)
        
            return return_list






class nodelist_Handler:

    def __init__(self,nodelist):
        self.nodelist=nodelist
        self.parentsId=set(self.filter("parent"))
        self.nodesId=set(self.filter("number"))
        self.leavesId=self.nodesId-self.parentsId
        self.leaves=self.filter(None,None,"number",self.leavesId)
        self.binary_variablesId=list(self.nodelist[0]("LPsol_bin").keys())
            

        return{}
           
    
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
            if(type(list(input_field_set[0])!=type(node_set[0][input_field]))):
                raise Exception("Input field set doesn't match the elements in nodeset for the specified field")

            if output_field!=None:
                filtered_nodelist=[l[output_field] for l in node_set if l[input_field] in input_field_set ]
            else:
                filtered_nodelist=[l for l in node_set if l[input_field] in input_field_set ]               

        return{filtered_nodelist}

    def is_a_Cover(self,nodeset):
        binary_dim=len(self.binary_variablesId)
        lbs=self.filter("lb",nodeset)
        ubs=self.filter("ub",nodeset)
        upper_coverage=0
        lower_coverage=0
        sticked_variables=[]
        lb_tilde={}
        ub_tilde={}
        for b in range(binary_dim):
            lb_tilde[b]=1
            ub_tilde[b]=0

            for l in self.nodesId:
                lb_tilde[b]=(min(lb_tilde[b],lbs[l][self.binary_variablesId[b]]))
                ub_tilde[b]=(max(ub_tilde[b],ubs[l][self.binary_variablesId[b]]))
        
            upper_coverage+=ub_tilde[b]
            lower_coverage+=lb_tilde[b]
            if ub_tilde[b]==lb_tilde[b]:
                sticked_variables.append(self.binary_variablesId[b])
    
        if(upper_coverage==binary_dim & lower_coverage==0):
            return{True,None}
        else:
            return{False,sticked_variables}
        
        
        return{}
    
    


        


    

    
