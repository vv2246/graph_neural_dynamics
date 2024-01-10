
from dynamics import Dynamics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import torchdiffeq
import warnings
# from utilities import set_seeds
from NeuralPsi import ODEFunc
import random
# import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def normalized_laplacian(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 * ( D - A ) * D^-1/2 = I - D^-1/2 * ( A ) * D^-1/2
    """
    # A =
    out_degree = A.sum(1)#np.array(A.sum(1), dtype=np.float32)
    in_degree = A.sum(0)#np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = torch.pow(out_degree,-0.5)#np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = torch.pow(in_degree,-0.5)#np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = torch.eye(A.shape[0]) - torch.diag(out_degree_sqrt_inv) @ A @ torch.diag(int_degree_sqrt_inv)
    return mx_operator

class GCN(nn.Module):

    def __init__(self, A, d, h, bias = True):
        super(GCN, self).__init__()

        self.A = A
        self.A_prop = torch.FloatTensor(normalized_laplacian(A))
        self.g1 = nn.Sequential(nn.Linear(d, h), nn.Tanh())
        self.g2 = nn.Sequential(GraphConvolution(h,h,self.A_prop),nn.Tanh())
        self.g3 = nn.Sequential(GraphConvolution(h,h ,self.A_prop),nn.Tanh())
        # self.g3 = nn.Sequential(GraphConvolution(h,d,self.A_prop),nn.Tanh())
        self.g4 = nn.Sequential(nn.Linear(h, h), nn.Tanh() ,nn.Linear(h, d))
        
    def forward(self, t, x):
        x = x[:,:,0]
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = self.g4(x)
        return x[:,None]
    

class GraphConvolution(nn.Module):

    def __init__(self, input_size, output_size,A, bias=True):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=bias)
        self.A = A
    def forward(self, x):
        x = self.fc(x)
        output = torch.mm(self.A, x)
        return output#.view(1, -1)



    
class SimpleFullNN(nn.Module):
    """Maps R^n -> R^n, with overparameterized model."""

    def __init__(self, n, h, my_seed = 0):
        super(SimpleFullNN, self).__init__()

        self.sigma = nn.Tanh()
        self.seed = my_seed
        torch.cuda.manual_seed_all(self.seed)

        self.g = nn.Sequential(
            nn.Linear(n, h),
            self.sigma,
            nn.Linear(h, h),
            self.sigma,
            nn.Linear(h, h),
            self.sigma,
            nn.Linear(h, n),
        )

    def forward(self,t, x):
        x = x[:,:,0].T
        out = self.g(x)
        return out.T[:,None] #torch.transpose(out,0,1)


if __name__ == "__main__":
    
    # graph
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        
    
    g = nx.Graph()
    g.add_edge(0,1)
    # nx.draw(g)
    # plt.show()
    model = "SIS"
    # nn_model = "ODE"
    A = torch.FloatTensor(np.array(nx.adjacency_matrix(g).todense()))
    
    # dynamics
    if model == "MAK":
        dyn = Dynamics(A=A, B= 0.1, R= 1, F=0.5, b=1,  model = model)
    if model == "SIS":
        dyn = Dynamics(A=A, B= 4, R =5,  model = model)
    if model == "Diffusion":
        dyn = Dynamics(A=A, B= 0.5,  model = model)
    if model == "PD":
        dyn = Dynamics(A=A, B= 2, R= 0.3, a= 1.5, b= 3,  model = model)
    if model == "MM":
        dyn = Dynamics(A=A, B= 1, R = 1, H=3,  model = model)
    
    train_distr = torch.distributions.Beta(torch.FloatTensor([5]),torch.FloatTensor([2]))
    y_train = []
    x_train = []
    size = 2
    train_samples = 200
    for i in range(train_samples):
        train_distr.sample([size]).to(device)
        x0 = train_distr.sample([size]).to(device)
        y_train.append(dyn(0, x0))
        x_train.append(x0)
    
    ### plot time series 
    x0 = torch.FloatTensor([0,1])[:,None]
    T=15
    time_tick= 500
    t = torch.linspace(0., T, time_tick)
    sol = torchdiffeq.odeint(dyn, x0, t, method='dopri5').squeeze().t()
    # plt.plot(sol.T)
    
    # Ensure x and y are meshgrid outputs
    x = torch.linspace(0, 2, 10)
    y = torch.linspace(0, 2, 10)
    X, Y = torch.meshgrid(x, y, indexing = 'xy')
    
    # Flatten the meshgrid arrays for processing
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    xy = torch.stack((X_flat, Y_flat))
    
    # Compute the vector field
    g = dyn(0, xy)
    Fx = np.array(g[0,:]).reshape(X.shape)
    Fy = np.array(g[1,:]).reshape(Y.shape)
    
    
    ###### MODELS
    warnings.filterwarnings('ignore')
    
    
    for nn_model in ["GCN", "NeuralPsi","NN"]:
        if nn_model == "NeuralPsi":
            func = ODEFunc(A = A, d = 1, h=30, h2=30, h3 = 30, h4 = 30, 
                            bias = True,  Q_factorized= True).to(device)
        elif nn_model == "NN":
            func = SimpleFullNN(2, 60)
        elif nn_model == "GCN":
            func = GCN(A = A, d= 1, h = 30).to(device)
    
        ## training 
        optimizer = torch.optim.Adam(func.parameters(), lr=0.01, weight_decay=0)
            
        for itr in range(501):
            optimizer.zero_grad()
            pred_y = [func(0, x_train[i][:,None]) for i in range(train_samples)]
            
            v1= torch.squeeze(torch.cat(pred_y,1),2)
            v2 = torch.cat(y_train,1)
            loss =torch.abs(v1-v2).mean()
            
            loss.backward()
            optimizer.step()
            if itr % 100 == 0:
                print(nn_model,itr, loss)
            
        
        g_pred = torch.vstack([func(0,xy[:,i][:,None,None]).detach().squeeze() for i in range(xy.shape[1])]).T
        Fx_pred = np.array(g_pred[0,:]).reshape(X.shape)
        Fy_pred = np.array(g_pred[1,:]).reshape(Y.shape)
            
        fig, ax = plt.subplots()
        ax.streamplot(X.numpy(), Y.numpy(), Fx, Fy, color="royalblue", density=0.8, arrowstyle='->', arrowsize=1.5)
        ax.streamplot(X.numpy(), Y.numpy(), Fx_pred, Fy_pred, color="hotpink", density=0.8, arrowstyle='->', arrowsize=1.5)
        ax.scatter(torch.stack(x_train).T.squeeze()[0,:],torch.stack(x_train).T.squeeze()[1,:], s= 5, c= "k")
        ax.set_ylim(0,2)
        ax.set_xlim(0,2)
        fig.suptitle(model+ " "+ nn_model)
        plt.show()


    
    