from dynamics import Dynamics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import warnings
from NeuralPsi import ODEFunc
import random
from tqdm import tqdm
from utilities import save_results, ModelParameters, GCN, GCN_single
from torch_geometric.utils import from_networkx
    

    
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
        return out.T[:,None] 


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
    model = "MAK"
    A = torch.FloatTensor(np.array(nx.adjacency_matrix(g).todense())).to(device)
    
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
    y_train = []
    x_train = []
    size = 2
    number_train_samples = 200
    train_distr = torch.distributions.Beta(torch.FloatTensor([5]),torch.FloatTensor([2]))
    train_samples = [train_distr.sample([size]).to(device) for i in range(number_train_samples)]
    # train_samples = 200
    for i in range(number_train_samples):
        train_distr.sample([size]).to(device)
        # x0 = train_distr.sample([size]).to(device)
        x0 = train_samples[i].to(device)
        y_train.append(dyn(0, x0))
        x_train.append(x0)
    
    
    ###### MODELS
    warnings.filterwarnings('ignore')
    nn_model = "NN"
    
    func = SimpleFullNN(2, 60).to(device)
    ## training 
    optimizer = torch.optim.Adam(func.parameters(), lr=0.005, weight_decay=0.)
    epochs = 1000

    for itr in tqdm(range(epochs)):
        optimizer.zero_grad()
        pred_y = [func(0, x_train[i][:,None]) for i in range(number_train_samples)]
        
        v1= torch.squeeze(torch.cat(pred_y,1),2)
        v2 = torch.cat(y_train,1)
        loss =torch.abs(v1-v2).mean()
        
        loss.backward()
        optimizer.step()
        if itr % 100 == 0:
            print(nn_model,itr, loss)
        
    # Ensure x and y are meshgrid outputs
    x = torch.linspace(0, 2, 10)
    y = torch.linspace(0, 2, 10)
    X, Y = torch.meshgrid(x, y, indexing = 'xy')
    
    # Flatten the meshgrid arrays for processing
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    xy = torch.stack((X_flat, Y_flat)).to(device)
    
    # Compute the vector field
    g = dyn(0, xy)
    Fx = np.array(g[0,:].cpu()).reshape(X.shape)
    Fy = np.array(g[1,:].cpu()).reshape(Y.shape)
    
    
    g_pred = torch.vstack([func(0,xy[:,i][:,None,None]).detach().squeeze() for i in range(xy.shape[1])]).T
    Fx_pred = np.array(g_pred[0,:].cpu()).reshape(X.shape)
    Fy_pred = np.array(g_pred[1,:].cpu()).reshape(Y.shape)
        
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize = (5,5))
    ax.streamplot(X.numpy(), Y.numpy(), Fx, Fy, color="royalblue", density=0.8, arrowstyle='->', arrowsize=1.5, zorder = 0)
    ax.streamplot(X.numpy(), Y.numpy(), Fx_pred, Fy_pred, color="hotpink", density=0.8, arrowstyle='->', arrowsize=1.5,zorder =0)
    ax.set_ylim(0,2)
    ax.set_xlim(0,2)
    ax.set_ylabel("$x_2$")
    ax.set_xlabel("$x_1$")
    ax.scatter(torch.hstack(x_train)[0],torch.hstack(x_train)[1], c = "k", marker = "x", zorder =1 ,linewidths = 0.8)
    ax.set_title(model + " "+ nn_model )
    ax.set_yticks(np.linspace(0,2,6))
    ax.set_xticks(np.linspace(0,2,6))
    plt.tight_layout()
    plt.savefig(f"figures/ffnn_{model}.pdf")

    
    