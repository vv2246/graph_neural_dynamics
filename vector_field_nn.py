
from dynamics import Dynamics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
# import torchdiffeq
import warnings
# from utilities import set_seeds
from NeuralPsi import ODEFunc
import random
from tqdm import tqdm

    
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
    
    
    ###### MODELS
    warnings.filterwarnings('ignore')
    nn_model = "NN"
    
    func = SimpleFullNN(2, 60).to(device)
    ## training 
    optimizer = torch.optim.Adam(func.parameters(), lr=0.01, weight_decay=0.)
    epochs = 500

    for itr in tqdm(range(epochs)):
        optimizer.zero_grad()
        pred_y = [func(0, x_train[i][:,None]) for i in range(train_samples)]
        
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
        
    fig, ax = plt.subplots()
    ax.streamplot(X.numpy(), Y.numpy(), Fx, Fy, color="royalblue", density=0.8, arrowstyle='->', arrowsize=1.5)
    ax.streamplot(X.numpy(), Y.numpy(), Fx_pred, Fy_pred, color="hotpink", density=0.8, arrowstyle='->', arrowsize=1.5)
    ax.scatter(torch.stack(x_train).T.squeeze()[0, :].cpu(), torch.stack(x_train).T.squeeze()[1, :].cpu(), s=5, c="k")
    ax.set_ylim(0,2)
    ax.set_xlim(0,2)
    fig.suptitle(model+ " "+ nn_model + f", error: {round(float(torch.abs(g_pred - g).mean()),2)}")
    plt.show()


    
    