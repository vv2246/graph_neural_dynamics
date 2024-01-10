
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
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx




class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,add_self_loops=True, normalize= True)
        self.conv2 = GCNConv(hidden_channels, out_channels,add_self_loops=True, normalize= True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.tanh(x)
        x = self.conv2(x, edge_index)
        # print(x)
        return F.tanh(x)#, dim=1)

if __name__ == "__main__":
    
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # graph
    g = nx.Graph()
    g.add_edge(0,1)
    # nx.draw(g)
    # plt.show()
    model = "Diffusion"
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
    training_data = []
    for i in range(train_samples):
        train_distr.sample([size]).to(device)
        x0 = train_distr.sample([size]).to(device) # features 
        y0 = dyn(0, x0) # labels 
        sample_data = from_networkx(g)
        sample_data.x = x0
        sample_data.y = y0
        training_data.append(sample_data)
        y_train.append(dyn(0, x0))
        x_train.append(x0)
    
    ### plot time series 
    # x0 = torch.FloatTensor([0,1])[:,None]
    # T=15
    # time_tick= 500
    # t = torch.linspace(0., T, time_tick)
    # sol = torchdiffeq.odeint(dyn, x0, t, method='dopri5').squeeze().t()
    # plt.plot(sol.T)
    
    # # Ensure x and y are meshgrid outputs
    # x = torch.linspace(0, 2, 10)
    # y = torch.linspace(0, 2, 10)
    # X, Y = torch.meshgrid(x, y, indexing = 'xy')
    
    # # Flatten the meshgrid arrays for processing
    # X_flat = X.flatten()
    # Y_flat = Y.flatten()
    # xy = torch.stack((X_flat, Y_flat))
    
    # # Compute the vector field
    # g = dyn(0, xy)
    # Fx = np.array(g[0,:]).reshape(X.shape)
    # Fy = np.array(g[1,:]).reshape(Y.shape)
    
    
    # ###### MODELS
    warnings.filterwarnings('ignore')
    func = GCN(in_channels = 1, hidden_channels=30, out_channels= 1).to(device)
    
    
    # for nn_model in ["GCN", "NeuralPsi","NN"]:
    #     if nn_model == "NeuralPsi":
    #         func = ODEFunc(A = A, d = 1, h=30, h2=30, h3 = 30, h4 = 30, 
    #                         bias = True,  Q_factorized= True).to(device)
    #     elif nn_model == "NN":
    #         func = SimpleFullNN(2, 60)
    #     elif nn_model == "GCN":
    #         func = GCN(A = A, d= 1, h = 30).to(device)
    
    ## training 
    optimizer = torch.optim.Adam(func.parameters(), lr=0.01, weight_decay=0)
    # l =func(sample_data)
    for itr in range(501):
    #     # print(itr,"... ")
        optimizer.zero_grad()
        pred_y = [func(d) for d in training_data]
            
        v1 = torch.cat(pred_y,1) #torch.squeeze(torch.cat(pred_y,1),2)
        v2 = torch.cat(y_train,1)
        loss =torch.abs(v1-v2).mean()
            
        loss.backward()
        optimizer.step()
        if itr % 100 == 0:
            print(itr, loss)
            
        
    #     g_pred = torch.vstack([func(0,xy[:,i][:,None,None]).detach().squeeze() for i in range(xy.shape[1])]).T
    #     Fx_pred = np.array(g_pred[0,:]).reshape(X.shape)
    #     Fy_pred = np.array(g_pred[1,:]).reshape(Y.shape)
            
    #     fig, ax = plt.subplots()
    #     ax.streamplot(X.numpy(), Y.numpy(), Fx, Fy, color="royalblue", density=0.8, arrowstyle='->', arrowsize=1.5)
    #     ax.streamplot(X.numpy(), Y.numpy(), Fx_pred, Fy_pred, color="hotpink", density=0.8, arrowstyle='->', arrowsize=1.5)
    #     ax.scatter(torch.stack(x_train).T.squeeze()[0,:],torch.stack(x_train).T.squeeze()[1,:], s= 5, c= "k")
    #     ax.set_ylim(0,2)
    #     ax.set_xlim(0,2)
    #     fig.suptitle(model+ " "+ nn_model)
    #     plt.show()


    
    