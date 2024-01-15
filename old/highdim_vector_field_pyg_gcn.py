
from dynamics import Dynamics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import warnings
import random
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv
from torch_geometric.utils import from_networkx


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_name = "GCNConv"):
        super().__init__()
        
        if model_name == "GCNConv": 
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
        elif model_name == "ChebConv":
            self.conv1 = ChebConv(in_channels, hidden_channels, K = 10) # what is a good value for K????
            self.conv2 = ChebConv(hidden_channels, hidden_channels, K = 10)
            self.conv3 = ChebConv(hidden_channels, hidden_channels, K = 10)
        elif model_name == "SAGEConv":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
            
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.tanh(x)
        x = self.conv2(x, edge_index)
        x = torch.tanh(x)
        x = self.conv3(x, edge_index)
        x = torch.tanh(x)
        x = self.lin(x)
        return x 

if __name__ == "__main__":
    
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # graph
    G = nx.Graph()
    G.add_edge(0,1)
    # nx.draw(g)
    # plt.show()
    model_name = "ChebConv"
    model = "MAK"
    A = torch.FloatTensor(np.array(nx.adjacency_matrix(G).todense()))
    
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
    # y_train = []
    x_train = []
    size = 2
    train_samples = 200
    training_data = []
    for i in range(train_samples):
        train_distr.sample([size]).to(device)
        x0 = train_distr.sample([size]).to(device) # features 
        y0 = dyn(0, x0) # labels 
        sample_data = from_networkx(G)
        sample_data.x = x0
        sample_data.y = y0
        training_data.append(sample_data)
        # y_train.append(dyn(0, x0))
        x_train.append(x0)
    
    warnings.filterwarnings('ignore')
    func = GCN(in_channels = 1, hidden_channels=30, out_channels= 1, model_name = model_name).to(device)
    
    ## training 
    optimizer = torch.optim.Adam(func.parameters(), lr=0.01, weight_decay=0)
    for itr in range(501):
        optimizer.zero_grad()
        
        pred_y = [func(d) for d in training_data]
        true_y = [d.y for d in training_data]
            
        v1 = torch.cat(pred_y,1) 
        v2 = torch.cat(true_y,1)
        loss =torch.abs(v1-v2).mean()
            
        loss.backward()
        optimizer.step()
        if itr % 100 == 0:
            print(itr, loss)
            
            
        
    with torch.no_grad():
        test_with_new_graph = True
        if test_with_new_graph :
            # take new graph
            test_size = 12
            G_test = nx.erdos_renyi_graph(test_size, 0.5)
            A_test = torch.FloatTensor(np.array(nx.adjacency_matrix(G_test).todense()))
        else: 
            test_size = size
            A_test = A
            G_test = G
        
        # Ensure x and y are meshgrid outputs
        x = torch.linspace(0, 2, 10)
        y = torch.linspace(0, 2, 10)
        X, Y = torch.meshgrid(x, y, indexing = 'xy')
        
        # # Flatten the meshgrid arrays for processing
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        # xy = torch.stack((X_flat, Y_flat))
        x_rest = train_distr.sample([test_size - 2])
        x_rest = x_rest.repeat(1, Y_flat.shape[0] )
        y_rest = train_distr.sample([test_size - 2])
        y_rest = y_rest.repeat(1, Y_flat.shape[0] )
        xy = torch.vstack([X_flat, Y_flat ,y_rest])
        
        # dynamics
        if model == "MAK":
            dyn_test = Dynamics(A=A_test, B= 0.1, R= 1, F=0.5, b=1,  model = model)
        if model == "SIS":
            dyn_test = Dynamics(A=A_test, B= 4, R =5,  model = model)
        if model == "Diffusion":
            dyn_test = Dynamics(A=A_test, B= 0.5,  model = model)
        if model == "PD":
            dyn_test = Dynamics(A=A_test, B= 2, R= 0.3, a= 1.5, b= 3,  model = model)
        if model == "MM":
            dyn_test = Dynamics(A=A_test, B= 1, R = 1, H=3,  model = model)
        
        # Compute the vector field
        g = dyn_test(0, xy)
        Fx = np.array(g[0,:]).reshape(X.shape)
        Fy = np.array(g[1,:]).reshape(Y.shape)
        
        
        func.eval()
        g_pred = []
        for i in range(xy.shape[1]):
            data = from_networkx(G_test)
            data.x = xy[:,i][:,None]
            g_pred.append(func(data))
            
        g_pred = torch.stack(g_pred).squeeze().T
        Fx_pred = np.array(g_pred[0,:]).reshape(X.shape)
        Fy_pred = np.array(g_pred[1,:]).reshape(Y.shape)
        
        
        fig, ax = plt.subplots()
        ax.streamplot(X.numpy(), Y.numpy(), Fx, Fy, color="royalblue", density=0.8, arrowstyle='->', arrowsize=1.5)
        ax.streamplot(X.numpy(), Y.numpy(), Fx_pred, Fy_pred, color="hotpink", density=0.8, arrowstyle='->', arrowsize=1.5)
        # ax.scatter(torch.stack(x_train).T.squeeze()[0,:],torch.stack(x_train).T.squeeze()[1,:], s= 5, c= "k")
        ax.set_ylim(0,2)
        ax.set_xlim(0,2)
        fig.suptitle(model+ " GNN " + model_name+ f", error: {round(float(torch.abs(g_pred - g).mean()),2)}")
        plt.show()

        
    