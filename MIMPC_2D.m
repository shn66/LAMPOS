function [sol, x_sol, u_sol, b_sol]= MIMPC_2D(x0,ol,obs_size) 

% Fixed parameters
N_obs_max=10;
M=20;


% Model data
A = [1 0 0.1 0;
     0 1 0 0.1;
     0 0 1 0;
     0 0 0 1];
B = 0.1*[0 0;0 0;1 0;0 1];
nx = size(A,1); % Number of states
nu = size(B,2); % Number of inputs

% MPC data
Q = 100*eye(nx);
R = 10*eye(nu);
N = 20;

lb_x=[-15;-15;-10;-10];
ub_x=-lb_x;

lb_u=[-5;-5];
ub_u=-lb_u;

% If no arguments are passed to the function, default intialisation below
if nargin<2
    x0=[3; 0;0;0];

    ol=[1 1 3 5 7 9 10;
        3 -1 5 5 5 5 5];

    obs_size=[1 1 1 1 1 1 2;
              2 2 2 2 2 2 1];
end


n_obs=size(ol,2);

if n_obs<N_obs_max
    ol=[ol M*ones(2,N_obs_max-n_obs)];
    obs_size=[obs_size zeros(2,N_obs_max-n_obs)];
end
ou=ol+obs_size;





% Decision variables
x = sdpvar(nx,N+1);
u = sdpvar(nu,N);

b_l=binvar(2,N,N_obs_max);b_u=binvar(2,N,N_obs_max);

s_x_l=sdpvar(nx,N);s_x_u=sdpvar(nx,N);
s_u_l=sdpvar(nu,N);s_u_u=sdpvar(nu,N);

s_oa_l=sdpvar(2,N,N_obs_max);s_oa_u=sdpvar(2,N,N_obs_max);

s_oa_side=sdpvar(1,N,N_obs_max);


% MPC problem setup

constraints = [x(:,1)==x0; s_x_l>=0; s_x_u>=0; s_u_l>=0; s_u_u>=0;
               s_oa_l>=0; s_oa_u>=0; s_oa_side>=0];
objective   = 0;
for k = 1:N
 objective = objective + norm(Q*x(:,k),1) + norm(R*u(:,k),1);

 Model = [x(:,k+1) == A*x(:,k) + B*u(:,k)];
 
 obs_avoid=[];
 for i=1:N_obs_max
     obs_avoid=[obs_avoid; 
                x(1:2,k+1)+s_oa_l(:,k,i)==ol(:,i)+M*b_l(:,k,i);
                x(1:2,k+1)-s_oa_u(:,k,i)==ou(:,i)-M*b_u(:,k,i);
                b_l(1,k,i)+b_l(2,k,i)+b_u(1,k,i)+b_u(2,k,i)+s_oa_side(1,k,i)==2*2-1];
 end
 
 state_constr=[lb_x==x(:,k+1)-s_x_l(:,k);
                ub_x==x(:,k+1)+s_x_u(:,k)];
 
 input_constr=[lb_u==u(:,k)-s_u_l(:,k);
                ub_u==u(:,k)+s_u_u(:,k)];
 

 constraints = [constraints, Model, obs_avoid, state_constr, input_constr];
end

objective = objective + 100*norm(Q*x(:,k+1),1);
sol = optimize(constraints, objective,sdpsettings('verbose',0, 'solver', 'gurobi', 'savesolveroutput',1));

x_sol=value(x);
u_sol=value(u);
b_sol=[value(b_l);value(b_u)];
end