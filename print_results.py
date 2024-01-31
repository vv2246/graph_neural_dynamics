from utilities import load_results
from dynamics import Dynamics
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import from_networkx
import networkx as nx

import warnings
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
warnings.filterwarnings('ignore')
for dynamics_name in ["Diffusion","MAK", "MM","PD","SIS"]:#
    print("---", dynamics_name , "---")
    # for model in ["NeuralPsi", "SAGEConv","GraphConv","ResGatedGraphConv","GATConv","ChebConv"]:
    for model in ["SAGEConv_single",   "GraphConv_single", "ResGatedGraphConv_single", "GATConv_single","ChebConv_single" ]:
        A, params, func, x_train,y_train  = load_results(f"models/neural_network_{model}_dynamics_{dynamics_name}_graph_size_10", device)

        pytorch_total_params = sum(p.numel() for p in func.parameters() if p.requires_grad)
        # if "single" in model:
        #     print( " & " , "\\cmark", " & ", pytorch_total_params , end  = " & " )
        # else:
        print(model , " & ", pytorch_total_params , end  = " & " )
        for (test_with_new_graph, test_dist) in [(False, False), (False, True), (True, False)]:
            with torch.no_grad():
                if test_with_new_graph :
                    # take new graph
                    test_size = 20
                    G_test = nx.erdos_renyi_graph(test_size, 0.5)
                    A_test = torch.FloatTensor(np.array(nx.adjacency_matrix(G_test).todense())).to(device)
                else: 
                    test_size = params.size
                    A_test = A.to(device)
                    G_test = nx.from_numpy_array(np.array(A.cpu()))
                    
                if test_dist == True:
                    test_distr= torch.distributions.Uniform(0.5,1.5)#torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1])) + 
                    xy = test_distr.sample([test_size, 500 ]).squeeze().to(device)
                else:
                    if test_with_new_graph :
                        xy = params.train_distr.sample([test_size, 500 ]).squeeze().to(device)
                    else:
                        xy = torch.hstack(x_train).to(device)
                
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
                    dyn_test = Dynamics(A=A_test, B= params.B, R =params.R,  model = params.dynamics_name)
                if params.dynamics_name == "Diffusion":
                    params.B = 0.5
                    dyn_test = Dynamics(A=A_test, B= params.B,   model = params.dynamics_name)
                if params.dynamics_name == "PD":
                    params.B = 2
                    params.R = 0.3
                    params.a = 1.5
                    params.b = 3
                    dyn_test = Dynamics(A=A_test, B= params.B, R= params.R, a= params.a, b= params.b,   model = params.dynamics_name)
                if params.dynamics_name == "MM":
                    params.H = 3
                    params.B = 4
                    params.R = 0.5
                    dyn_test = Dynamics(A=A_test, B= params.B, R = params.R, H=params.H,   model = params.dynamics_name)
                
                # Compute the vector field
                dyn_test = dyn_test.to(device)
                g = dyn_test(0, xy)
                
                if params.model_name == "NeuralPsi":
                    g_pred = torch.vstack([func(0,xy[:,i][:,None,None],adj = A_test).detach().squeeze() for i in range(xy.shape[1])]).T
                else:
                    g_pred = []
                    for i in range(xy.shape[1]):
                        data = from_networkx(G_test).to(device)
                        data.x = xy[:,i][:,None]
                        g_pred.append(func(data))
                    
                    g_pred = torch.stack(g_pred).squeeze().T     
            if (test_with_new_graph == True )and ( test_dist== False):
                print( round(float(torch.abs(g_pred - g).mean()),2) , end = "")
            else:
                print( round(float(torch.abs(g_pred - g).mean()),2) , end = " & ")
                
            
        print("\\\\")
