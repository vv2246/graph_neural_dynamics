from utilities import load_results
from dynamics import Dynamics
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import from_networkx
import networkx as nx


# graph
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
A, params, func, x_train,y_train  = load_results("models/neural_network_NeuralPsi_dynamics_MAK_graph_size_10")

test_with_new_graph = False
with torch.no_grad():
    if test_with_new_graph :
        # take new graph
        test_size = 12
        G_test = nx.erdos_renyi_graph(test_size, 0.5)
        A_test = torch.FloatTensor(np.array(nx.adjacency_matrix(G_test).todense()))
    else: 
        test_size = params.size
        A_test = A
        G_test = nx.from_numpy_array(np.array(A))
    
    # Ensure x and y are meshgrid outputs
    x = torch.linspace(0, 2, 10)
    y = torch.linspace(0, 2, 10)
    X, Y = torch.meshgrid(x, y, indexing = 'xy')
    
    # # Flatten the meshgrid arrays for processing
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    # xy = torch.stack((X_flat, Y_flat))
    x_rest = params.train_distr.sample([test_size - 2])
    x_rest = x_rest.repeat(1, Y_flat.shape[0] )
    y_rest = params.train_distr.sample([test_size - 2])
    y_rest = y_rest.repeat(1, Y_flat.shape[0] )
    xy = torch.vstack([X_flat, Y_flat ,y_rest])
    

    
    # dynamics
    if params.dynamics_name == "MAK":
        params.B = 0.1
        params.R = 1
        params.F = 0.5
        params.b =1
        dyn_test = Dynamics(A=A_test, B= params.B, R= params.R, F=params.F, b=params.b,  model = params.dynamics_name)
    if params.dynamics_name == "SIS":
        params.B = 4
        params.R = 5
        dyn_test = Dynamics(A=A_test, B= params.B, R =params.R,  model = params.dynamics.name)
    if params.dynamics_name == "Diffusion":
        params.B = 0.5
        dyn_test = Dynamics(A=A_test, B= params.B,   model = params.dynamics.name)
    if params.dynamics_name == "PD":
        params.B = 2
        params.R = 0.3
        params.a = 1.5
        params.b = 3
        dyn_test = Dynamics(A=A_test, B= params.B, R= params.R, a= params.a, b= params.b,   model = params.dynamics.name)
    if params.dynamics_name == "MM":
        dyn_test = Dynamics(A=A_test, B= params.B, R = params.R, H=params.H,   model = params.dynamics.name)
    
    # Compute the vector field
    g = dyn_test(0, xy)
    Fx = np.array(g[0,:]).reshape(X.shape)
    Fy = np.array(g[1,:]).reshape(Y.shape)
    
    
    if params.model_name == "NeuralPsi":
        g_pred = torch.vstack([func(0,xy[:,i][:,None,None],adj = A_test).detach().squeeze() for i in range(xy.shape[1])]).T
        Fx_pred = np.array(g_pred[0,:]).reshape(X.shape)
        Fy_pred = np.array(g_pred[1,:]).reshape(Y.shape)
    else:
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
ax.set_ylim(0,2)
ax.set_xlim(0,2)

fig.suptitle(params.dynamics_name+ " "+ params.model_name + f", error: {round(float(torch.abs(g_pred - g).mean()),2)}")
plt.show()
