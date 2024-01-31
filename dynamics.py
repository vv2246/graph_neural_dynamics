import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import networkx as nx
# import torch.nn.functional as F
import torchdiffeq

class Dynamics_individual(nn.Module):
    """
    Dynamics from Baruch-Barabasi paper + Diffusion
    """
    def __init__(self,  A, B = 1,  R = 1, H = 1, F = 1, a = 1, b = 2, model = "MAK", device = 'cpu'):
        super(Dynamics_individual, self ).__init__()
        self.B = B
        self.A = A
        self.R = R
        self.H = H
        self.F = F
        self.a = a
        self.b = b
        self.L =  torch.diag(A.sum(0)) - A
        self.model = model
        # self.self_interaction = self_interaction
        # self.nbr_interaction = nbr_interaction

    def forward(self, t, x):
        """
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if self.model == "MAK":
            # proteins are produced at a rate F, degraded at rate B , and generate hetero-dimers at rate R
            # F_self = self.F - self.B * x
            # F_nbr =  - x * torch.mm(self.A, x) * self. R
            
            # if debug__ :
            # f = F_self + F_nbr
            x_list_self = []
            x_list_nbr = []
            for i in range(len(x)):
                # xi_new =  - x[i]* sum([self.A[i,j] * x[j] for j in range(len(x))]) * self.R
                x_list_self.append(self.F - self.B * x[i])
                x_list_nbr.append( [- x[i] * self.A[i,j] * x[j] * self.R for j in range(len(x))])
                # for j in range(len(x)):
                    # print( x[i],self.A[j,i],  x[j])
                # print( torch.Tensor(x_list)[:, None] - f )
            # print(x_list_nbr)
                
        if self.model == "PD":
            # describes the population density at site i.
            # The first term describes the local population dynamics and the second 
            # term describes the coupling between adjacent sites
            F_self = - self.B * (x ** self.b) 
            F_nbr = torch.mm(self.A , (x ** self.a )) * self.R
            
        if self.model == "MM":
            # dynamics of gene regulation. 
            F_self = -self.B * x 
            F_nbr = self.R * torch.mm(self.A, x ** self.H / (x ** self.H + 1))
            
        if self.model == "SIS":
            F_self = - self.B * x
            F_nbr = (1 - x ) *  torch.mm(self.A , x) * self.R 
            
        if self.model =="Diffusion":
            F_self = 0 * x 
            F_nbr =  - torch.mm(self.L, x) * self.B
        
        # f = 0
        # if self.self_interaction == True:
        #     f += F_self
        # if self.nbr_interaction == True: 
        #     f += F_nbr
        return  (torch.tensor(x_list_self)), torch.tensor(x_list_nbr) #x_list_self, x_list_nbr 


class Dynamics(nn.Module):
    """
    Dynamics from Baruch-Barabasi paper + Diffusion
    """
    def __init__(self,  A, B = 1,  R = 1, H = 1, F = 1, a = 1, b = 2, model = "MAK", self_interaction = True, nbr_interaction = True, device = 'cpu'):
        super(Dynamics, self ).__init__()
        self.B = B
        self.A = A
        self.R = R
        self.H = H
        self.F = F
        self.a = a
        self.b = b
        self.L =  torch.diag(A.sum(0)) - A
        self.model = model
        self.self_interaction = self_interaction
        self.nbr_interaction = nbr_interaction
        self.in_degree = A.sum(0)[:,None]

        if model == "RO":
            
            self.epsilon = 0.1
            self.a = 0.35
            self.b = 0.2
            self.c = -5.7

    def forward(self, t, x, debug__= False):
        """
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if self.model == "MAK":
            # proteins are produced at a rate F, degraded at rate B , and generate hetero-dimers at rate R
            F_self = self.F - self.B * (x ** self.b)
            F_nbr =  - x * torch.mm(self.A, x) * self. R
            
                
        if self.model == "PD":
            # describes the population density at site i.
            # The first term describes the local population dynamics and the second 
            # term describes the coupling between adjacent sites
            F_self = - self.B * (x ** self.b) 
            F_nbr = torch.mm(self.A , (x ** self.a )) * self.R
            
        if self.model == "MM":
            # dynamics of gene regulation. 
            F_self = -self.B * x 
            F_nbr = self.R * torch.mm(self.A, x ** self.H / (x ** self.H + 1))
            # print(F_nbr)
            
        if self.model == "SIS":
            F_self = - self.B * x
            F_nbr = (1 - x ) *  torch.mm(self.A , x) * self.R 
            
        if self.model =="Diffusion":
            F_self = 0 * x 
            F_nbr =  - torch.mm(self.L, x) * self.B
            
            
        if self.model == "RO":
            x1,x2,x3 = x[:,0][:,None], x[:,1][:,None], x[:,2][:,None]
            x1j = x[:,0][:,None]
            F_nbr = torch.mm(self.L, x1j) 
            F1 = - x2 - x3 + F_nbr * self.epsilon
            F2 = x1 + self.a* x2
            F3 = self.b + x3*(x1 + self.c)
            f = torch.vstack([F1.squeeze(),F2.squeeze(),F3.squeeze()]).T
            
            # x1 = x[:, 0:1]
            # F_nbr = torch.mm(self.L, x1) 
            # F1 = -x[:, 1:2] - x[:, 2:3] + F_nbr * self.epsilon
            # F2 = x1 + self.a * x[:, 1:2]
            # F3 = self.b + x[:, 2:3] * (x1 + self.c)
            # f = torch.cat([F1, F2, F3], dim=1)
            
        if self.model =="FHN":
            epsilon = 1
            a = 0.28 
            b = 0.5 
            c = -0.04
            x1,x2 = x[:,0][:,None], x[:,1][:,None]
            x1j = x[:,0][:,None]
            F_nbr = - torch.mm(self.L, x1j) / self.in_degree
            F1 = x1 - x1**3 - x2 - epsilon * F_nbr
            # print(F1.shape )
            F2 = a + b * x1 + c * x2
            # print(F1.shape, F2.shape, self.in_degree.shape)
            f = torch.vstack([F1.squeeze(),F2.squeeze()]).T
            # print(f.shape)
            
            
        if self.model == "HR":
            Iext = 3.24 # external current 
            Vsyn = 2
            Lambda = 10
            Omegasyn = 1
            c = 1
            u = 5
            r = 0.005
            x0 = -1.6
            epsilon = 0.30
            # epsilon2 = -0.15
            a = 1
            b = 3
            s=4
            x1,x2,x3 = x[:,0][:,None], x[:,1][:,None], x[:,2][:,None]
            x1j = x[:,0][:,None]
            mu = (1+ torch.exp(-(Lambda * (x1j - Omegasyn ))))**(-1)
            F_nbr =  epsilon * (Vsyn - x1) * torch.mm(self.A, mu)
            
            F1 = x2 - a* (x1**3) + b * (x1**2) - x3  + Iext + F_nbr
            F2 = c - u * (x1**2) - x2
            F3 = r* ( s* (x1-x0) - x2)
            f = torch.vstack([F1.squeeze(),F2.squeeze(),F3.squeeze()]).T
        
        if self.model not in [ "HR" ,"RO","FHN"]:
            f = 0
            if self.self_interaction == True:
                f += F_self
            if self.nbr_interaction == True: 
                f += F_nbr
        return f


class Lorenz(nn.Module):
    # Lorenz system 
    def __init__(self,rho=10,sigma=10, beta= 8/3):
        super(Lorenz,self).__init__()
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
    def forward(self, t, vec):
        x,y,z= vec[0]
        dxdt = self.sigma*(y-x)
        dydt = x*(self.rho-z) -y
        dzdt = x*y - self.beta*z
        return torch.Tensor([[dxdt, dydt, dzdt]])


     
class ChaoticDynamics(nn.Module):
    #from Sprot 2008
    def __init__(self,  A,b):
        super(ChaoticDynamics, self ).__init__()
        self.b = b
        self.A = A
        

    def forward(self, t, x):
        """
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        return torch.tanh( torch.mm(self.A, x)) - self.b * x
        

    
if __name__ == "__main__":

    import numpy as np
    n=3

    A = torch.FloatTensor(np.matrix('0 1 0; 0 0 1;1 1 0'))
    b = torch.FloatTensor(np.matrix('0.0043;0.0043;0.0043;0.0043'))
    dyn = Dynamics(A=A, model  = "FHN")
    # dyn = Dynamics_individual(A=A)
    x0 = torch.rand([n,2])
    
    ###dynamics
    T=60
    time_tick= 200
    t = torch.linspace(0., T, time_tick)
    
    dyn(0,x0)
    solution_numerical = torchdiffeq.odeint( dyn, x0, t, method="dopri5")
    
        
    plt.plot(solution_numerical[:,0,0])
    plt.plot(solution_numerical[:,0,1])
    # plt.plot(solution_numerical[:,0,2])
    plt.show()
    
    plt.plot(solution_numerical[:,1,0])
    plt.plot(solution_numerical[:,1,1])
    # plt.plot(solution_numerical[:,1,2])
    
    plt.show()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(solution_numerical[:, 0, 0].numpy(), solution_numerical[:,0, 1].numpy(), solution_numerical[:,0, 2].numpy())
    
    # print(f"numerical solution {solution_numerical.shape}")
    # plt.plot(solution_numerical.T)
    # plt.show()
    # plt.plot(solution_numerical[1,:], solution_numerical[0,:])
    # plt.show()
    
    
    