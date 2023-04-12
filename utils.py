import sys
sys.path.append("/home/mpc/LMILP/LAMPOS/")
from dataclasses import dataclass, field
import numpy as np
from typing import List
from pyscipopt import Eventhdlr, SCIP_EVENTTYPE
import cvxopt.modeling as cp
from cvxopt import solvers
from mosek import iparam
import os 
import datetime
import re
import psutil


# Defininig data structure for LP Subproblems
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
    def __eq__(self,other):
        return (self.b==other.b).all() and (self.lb==other.lb).all() and (self.ub==other.ub).all() \
            and (self.x==other.x).all() and (self.y==other.y).all() and (self.y==other.y).all() \
                and (self.lmb==other.lmbd).all() and (self.k==other.k).all() and (self.tau==other.tau).all() \
                    and self.cost==other.cost and self.feas==other.feas

# Defininig Data structure for  MILP problems
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

            self.cov_set.append(tuple([tuple(list(sp.lb.values())), tuple(list(sp.ub.values()))]))
            
            if sp.feas!="infeasible":
                self.feas_nodes.append(sp)
        self.cov_set_train=np.asarray(self.cov_set)   
        self.cov_set=set(self.cov_set)
          
    
    def __eq__(self,other):

        return (self.b==other.b).all() #and self.opt_sp==other.opt_sp and self.cov_set==other.cov_set and (self.x_opt==other.x_opt).all() and (self.y_opt==other.y_opt).all() and self.feas_nodes==other.feas_nodes

class LP_SP_Builder:
    '''
    Class for generic LP subproblem construction 
    '''
    def __init__(self):
        self.constraintList = []
        self.str2constr = {}
        self.constraint_dict={}
        self.objective=[]
        self.prob=[]

    # We use Mosek for solving the generic LP
    
    def solve__(self):
        self.prob=cp.op(self.objective,self.constraintList)
        solvers.options['show_progress'] = False
        solvers.options['mosek'] = {iparam.log: 0}   
        self.prob.solve(solver='mosek')
      
             
    def reset(self):
        self.variableList=[]
        self.constraintList = []
        self.str2constr = {}
        self.constraint_dict={}
        self.objective=[]
        self.prob=[]

    # Methods for setting the problem

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

    # Methods for recovering information on primal and dual variables

    def getConstrList(self):
        return self.constraintList
    
    def getConstrDict(self):
        return self.constraint_dict

    def getConstr(self, str_):
        return self.constraintList[self.str2constr[str_]]
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
            "x_sol":x_sol,
            "u_sol":u_sol,
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
    

    
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED,self)


# Class for handling the list of Nodes
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

   
    
    def recursive_make_cover(self,ref_sp): # Method for cover completions (See README.md in Offline Folder)
        binary_vars=self.binary_variablesId
        nodes=self.leaves
    
        lb_ref=ref_sp['lb']
        ub_ref=ref_sp['ub']
        ref_dim=sum(np.array([lb_ref[key]!=ub_ref[key] for key in lb_ref.keys()]))
   
        fixed_vars={key:lb_ref[key] for key in lb_ref.keys() if lb_ref[key]==ub_ref[key]}
        for k in fixed_vars.keys():
            lbs=self.filter("lb",nodes)
            ubs=self.filter("ub",nodes)
            lb_new=lb_ref.copy()
            ub_new=ub_ref.copy()
            lb_new[k]=1-fixed_vars[k] 
            ub_new[k]=1-fixed_vars[k]
            fixed_vars_new={key:lb_new[key] for key in lb_new.keys() if lb_new[key]==ub_new[key]}

            contained=False
            intersect=False
            contained_sps=[]
            if(len(nodes)>len(lbs)):
                print('attention')
            for l in range(len(nodes)):
                if (np.array(list(lbs[l].values()))<=np.array(list(lb_new.values()))).all() and (np.array(list(ubs[l].values()))>=np.array(list(ub_new.values()))).all():
                    contained=True
                    break
                sp_dim=sum(np.array([lbs[l][key]!=ubs[l][key] for key in lb_ref.keys()]))
                if sp_dim>=ref_dim:
                    fixed_vars_sp={key:lbs[l][key] for key in lbs[l].keys() if lbs[l][key]==ubs[l][key]}
                    shared_vars={key:fixed_vars_sp[key]==fixed_vars_new[key] for key in fixed_vars_new.keys() if key in fixed_vars_sp.keys()}
                    if shared_vars:
                        intersect=(np.array(list(shared_vars.values()))).all()
                        a=1
                    else:
                        intersect=True
                    if intersect:
                        break
                elif (np.array(list(lbs[l].values()))>=np.array(list(lb_new.values()))).all() and (np.array(list(ubs[l].values()))<=np.array(list(ub_new.values()))).all():
                    contained_sps.append(nodes[l])
                
            if not contained and not intersect:
                ## Remove these elements from nodelist
                for l in contained_sps:
                    nodes.remove(l)
                    
                   
                ## Define nodedict with lb_new, ub_new and add to node_list
                nodedict = {
                        "number": -2,
                        "type":None,
                        "leavesNodesNumber":None,
                        "LPsol_bin": None,
                        #"LPsol_full":LPsol_full,
                        "x_sol":None,
                        "u_sol":None,
                        "LPdual_sol":None,
                        "objval": None,
                        #"first": firstlp,
                        "parent": -1,
                        "node":None,
                        "age": 0,
                        "depth": -1,
                        "lb":lb_new,
                        "ub":ub_new,
                        "constraints": None,
                        "primalbound": None,
                        "dualbound": None,
                }
                nodes.append(nodedict)
                
                lb_ref=lb_new.copy(); ub_ref=ub_new.copy()
                lb_ref[k]=0; ub_ref[k]=1
                ref_sp = {
                        "number": -1,
                        "type":None,
                        "leavesNodesNumber":None,
                        "LPsol_bin": None,
                        #"LPsol_full":LPsol_full,
                        "x_sol":None,
                        "u_sol":None,
                        "LPdual_sol":None,
                        "objval": None,
                        #"first": firstlp,
                        "parent": -1,
                        "node":None,
                        "age": 0,
                        "depth": -1,
                        "lb":lb_ref,
                        "ub":ub_ref,
                        "constraints": None,
                        "primalbound": None,
                        "dualbound": None,
                }
                ## Create ref_sp node_dict with with lb_ref, ub_Ref
                ref_dim=sum(np.array([lb_ref[key]!=ub_ref[key] for key in lb_ref.keys()]))
                if ref_dim==(len(lb_ref.values())):
                    return nodes

                else:
                    nodes=self.recursive_make_cover(ref_sp=ref_sp)
                    break
                    
        return nodes

def get_recentdate_str(dataset_path,N,dt):
    
    files = os.listdir(dataset_path)
    integer=int(0.1)
    decimal=int(dt/0.1)
    dt_str=str(integer)+str(decimal)
    dates = []
    for file in files:
        match = re.search("MILP_data_points_"+dt_str+"_N"+str(N)+"_(.*).p", file)
        #match = re.search("MILP_data_points_dt_N20_(.*).p", file)
        if match:
            date_str = match.group(1)
            try:
              date = datetime.datetime.strptime(date_str,"%Y%m%d-%H%M%S")
              dates.append(date)
            except:
              print("File not found")
    dates.sort(reverse=True)
    recent_date_str=dates[0].strftime("%Y%m%d-%H%M%S")
    return recent_date_str,dt_str

def compute_solution_time(prediction_time,lb_time,ub_time,parallel_solve=False):
    # If we use serial solving then we apply  Amdahl's law
    if(not parallel_solve):
        efficiency_factor=0.9
        cores_num=psutil.cpu_count()
        parallelization_factor=(1-efficiency_factor)+efficiency_factor/cores_num
    else:
        parallelization_factor=1
    solution_time=(prediction_time+lb_time+ub_time)*parallelization_factor
    return solution_time

