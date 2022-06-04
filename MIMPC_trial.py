import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Rectangle

def MIMPC_2D( x0=np.array([10, 6.5,0,0]),ol=np.array([[1,1,3,5,7,9,10],[3,-1,5,5,2,5,5]]),obs_size=np.array([[1,1,1,1,1,1,2],[2,2,2,2,2,2,1]])): 
    
    #Fixed parameters
    N_obs_max=10
    M=20



    # Model data
    A = np.matrix([[1, 0, 0.1, 0],[0, 1, 0, 0.1],[0, 0, 1, 0],[0, 0, 0, 1]])
    B = 0.1*np.matrix([[0, 0],[0, 0],[1, 0],[0, 1]])

    nx = A.shape[1]#% Number of states
    nu = B.shape[1]#% Number of inputs

    # MPC data
    Q = 100*np.identity(nx)
    R = 10*np.identity(nu)
    N = 30

    lb_x=np.array([-15,-15,-8,-8])
    ub_x=-lb_x

    lb_u=np.array([-5,-5])
    ub_u=-lb_u

    # If no arguments are passed to the function, default intialisation below
   

    n_obs=ol.shape[1]

    if n_obs<N_obs_max:
        ol=np.concatenate((ol, M*np.ones((2,N_obs_max-n_obs))),axis=1)
        obs_size=np.concatenate((obs_size, np.zeros((2,N_obs_max-n_obs))),axis=1)
    
    ou=ol+obs_size



    # Decision variables
    x = cp.Variable((nx,N+1))
    u = cp.Variable((nu,N))

    s_x_l=cp.Variable((nx,N))
    s_x_u=cp.Variable((nx,N))
    s_u_l=cp.Variable((nu,N))
    s_u_u=cp.Variable((nu,N))

    #Slack and binary varibles definition
    s_oa_l=[cp.Variable((2,N)) for _ in range(N_obs_max)]
    s_oa_u= [cp.Variable((2,N)) for _ in range(N_obs_max)]
    s_oa_side=[cp.Variable((2,N)) for _ in range(N_obs_max)]
    b_l=[cp.Variable((2,N),boolean=True) for _ in range(N_obs_max)]
    b_u=[cp.Variable((2,N),boolean=True) for _ in range(N_obs_max)]
    
    # MPC problem setup
    Model=[]
    constraints = [x[:,1]==x0, s_x_l>=0, s_x_u>=0, s_u_l>=0, s_u_u>=0]
                
    objective   = 0
    
    for k in range(N):
        objective = objective + cp.norm(Q@x[:,k],1) + cp.norm(R@u[:,k],1)
        constraints += [x[:,k+1]== A@x[:,k] + B@u[:,k]]
        
        state_constr=[]
        input_constr=[]
        
        constraints+=[lb_x==x[:,k+1]-s_x_l[:,k],
                    ub_x==x[:,k+1]+s_x_u[:,k]]
    
        constraints+=[lb_u==u[:,k]-s_u_l[:,k],
                    ub_u==u[:,k]+s_u_u[:,k]]

        obs_avoid=[]            
        for i in range (N_obs_max):
            constraints+=[
                    # x[0:2,k+1]<=ol[:,i]+M*b_l[i][:,k],
                        x[0:2,k+1]+s_oa_l[i][:,k]==ol[:,i]+M*b_l[i][:,k],
                        x[0:2,k+1]-s_oa_u[i][:,k]==ou[:,i]-M*b_u[i][:,k],
                    #   x[0:2,k+1]>=ou[:,i]-M*b_u[i][:,k],    
                    #   b_l[i][0,k]+b_l[i][1,k]+b_u[i][0,k]+b_u[i][1,k]<=2*2-1]               
                        b_l[i][0,k]+b_l[i][1,k]+b_u[i][0,k]+b_u[i][1,k]+s_oa_side[i][0,k]==2*2-1,
                        s_oa_l[i]>=0, s_oa_u[i]>=0, s_oa_side[i]>=0]

        #constraints+= [Model, obs_avoid, state_constr, input_constr] 
    
    
    objective = objective + 100*cp.norm(Q@x[:,k+1],1)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(verbose=True, solver=cp.SCIP)

    print("Status",problem.status)
    print("The optimal value is ", problem.value)
    print("A solution x is ")
    x_sol=x.value
    u_sol=u.value
    b_l_sol=[b_l[i].value for i in range(N_obs_max)]
    b_u_sol=[b_u[i].value for i in range(N_obs_max)]

    return x_sol, u_sol, b_l_sol, b_u_sol, ol, obs_size
    #print(x.value)
    
    
    
[x_sol, u_sol, b_l_sol, b_u_sol, ol, obs_size]=MIMPC_2D()
print(x_sol)
fig=plt.figure()
n_obs=obs_size.shape[-1]
plt.plot(x_sol[0,:], x_sol[1,:], '--')
for i in range(n_obs):
    plt.gca().add_patch(Rectangle((ol[0,i],ol[1,i]),obs_size[0,i],obs_size[1,i]))
plt.show()
