# this class will have two method and thw following properties
# it will have a time vector t that will be incremented every time the function step is called
# a target that is usually 0 
# the first one is the step that given an input will return the next state for the system under analisys
# the second one will be collision check to see if at any time step
# the third one will check whether the state of the system is close or not to the goal
import sys
sys.path.append("/home/mpc/LMILP/LAMPOS/")
import numpy as np
class world():
    def __init__(self) -> None:
        self.t=0
        self.T_MAX=100
        self.A = np.array([[1, 0, 0.1, 0],[0, 1, 0, 0.1],[0, 0, 1, 0],[0, 0, 0, 1]])
        self.B = 0.1*np.array([[0, 0],[0, 0],[1, 0],[0, 1]])
        self.ol=np.array([[-2.5,-2.,1, 2 ],[-2.5 ,1,-2, 1.5 ]])
        self.obs_size=np.array([[2.5,2,2, 1],[1.5,2,2.5, 1.5]])
        
    def set_init(self,x0):
        self.t=0
        self.x=x0.reshape((-1,1))
    def step(self,u):
        self.t+=1
        self.x=self.A@self.x+self.B@u
        
        return self.check_collision(), self.check_goal()
    def check_collision(self):
        collision=False
        
        for i in range(self.ol.shape[1]):
            ol=self.ol[:,i]
            obs_size=self.obs_size[:,i]
            if (self.x[0:2]>ol).all() and (self.x[0:2]<ol+obs_size).all():
                collision=True
        return collision
    def check_goal(self):
        status=None
        if np.linalg.norm(self.x[0:2])<1.0:
            status="Yes"
        if self.t>self.T_MAX:
            status="Timed out"
        return status