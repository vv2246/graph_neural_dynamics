import torch
import os
from dataclasses import dataclass
import numpy as np
import torch
import networkx as nx
import pickle
import os
from NeuralPsi import ODEFunc
from dynamics import Dynamics
# from vector_field_neuralpsi import GCN
import torch.nn as nn
# from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, SimpleConv, ResGatedGraphConv, GraphConv,GATConv, GatedGraphConv
from torch_geometric.utils import from_networkx
import random

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
        elif model_name == "ResGatedGraphConv":
            self.conv1 = ResGatedGraphConv(in_channels, hidden_channels)
            self.conv2 = ResGatedGraphConv(hidden_channels, hidden_channels)
            self.conv3 = ResGatedGraphConv(hidden_channels, hidden_channels)
        elif model_name == "GatedGraphConv":
            self.conv1 = GatedGraphConv(in_channels, hidden_channels)
            self.conv2 = GatedGraphConv(hidden_channels, hidden_channels)
            self.conv3 = GatedGraphConv(hidden_channels, hidden_channels)
        elif model_name == "GraphConv":
            self.conv1 = GraphConv(in_channels, hidden_channels)
            self.conv2 = GraphConv(hidden_channels, hidden_channels)
            self.conv3 = GraphConv(hidden_channels, hidden_channels)
        elif model_name == "GATConv":
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
            self.conv3 = GATConv(hidden_channels, hidden_channels)
            
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
    
    

class GCN_single(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_name = "GCNConv_single"):
        super().__init__()
        
        if model_name == "GCNConv_single": 
            self.conv1 = GCNConv(in_channels, hidden_channels)
        elif model_name == "ChebConv_single":
            self.conv1 = ChebConv(in_channels, hidden_channels, K = 10) # what is a good value for K????
        elif model_name == "SAGEConv_single":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
        elif model_name == "ResGatedGraphConv_single":
            self.conv1 = ResGatedGraphConv(in_channels, hidden_channels)
        elif model_name == "GraphConv_single":
            self.conv1 = GraphConv(in_channels, hidden_channels)
        elif model_name == "GATConv_single":
            self.conv1 = GATConv(in_channels, hidden_channels)
        elif model_name == "GatedGraphConv_single":
            self.conv1 = GatedGraphConv(in_channels, hidden_channels)
                
            
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.tanh(x)
        x = self.lin(x)
        return x 

@dataclass
class ModelParameters:
    model_name: str
    dynamics_name: str
    train_distr: torch.distributions
    train_samples: int = 100
    size: int = 2
    epochs: int = 1000
    lr: float = 1e-2  # learning rate for training
    weight_decay: float = 1e-3  # weight decay for training
    h: int = 6 #self interaction embedding dim
    h2: int = 8 #nbr interaction embedding dim
    h3: int = 8 #nbr interaction embedding dim
    d: int = 1
    Q_factorized: bool = True
    bias: bool = True
    R: float = np.nan
    F: float = np.nan
    a: float = np.nan
    b: float = np.nan
    H: float = np.nan
    B: float = np.nan

def load_results(folder, device):

    # graph adjacency
    adj = torch.load(f'{folder}/adjacency_matrix.pt')
    
    # training config
    with open(f'{folder}/training_config.pkl', 'rb') as f:
        training_params = pickle.load(f)
    
    if training_params.model_name == "NeuralPsi":
        func = ODEFunc(A = adj, d= training_params.d, 
                    h = training_params.h, 
                    h2 = training_params.h2, bias = training_params.bias , Q_factorized = training_params.Q_factorized,
                    h3 = training_params.h3 ).to(device)
    elif "single" in training_params.model_name:
        func = GCN_single(in_channels = 1, hidden_channels=training_params.h, out_channels= 1, model_name = training_params.model_name).to(device)
        
    else:
        func = GCN(in_channels = 1, hidden_channels=training_params.h, out_channels= 1, model_name = training_params.model_name).to(device)
        
    func.load_state_dict(torch.load(folder +"/neural_network.pth"))
    
    x_train = torch.load(f"{folder}/training_data_x.pt")
    y_train = torch.load(f"{folder}/training_data_y.pt")
        
    
    return adj, training_params, func,x_train, y_train

def save_results(model, folder_name, adj,training_parameters , x_train, y_train):
    folder =f"{folder_name}"
    print("saving files to: " ,folder)
    os.makedirs(folder, exist_ok=True)
    
    # NN
    checkpoint = model.state_dict()
    torch.save(checkpoint, f'{folder}/neural_network.pth')

    # graph
    torch.save(adj, f'{folder}/adjacency_matrix.pt') 
    
    with open(f"{folder}/training_config.pkl","wb+") as f:
        pickle.dump(training_parameters, f)
        
        
    #data 
    # with open(f"{folder}/training_data.pkl","wb+") as f:
    #     pickle.dump([x_train, y_train], f)
    torch.save(torch.stack(x_train),f"{folder}/training_data_x.pt")
    torch.save(torch.stack(y_train),f"{folder}/training_data_y.pt")



def set_seeds():
    """Sets seeds for reproducibility."""
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)



def compute_d_statistics_one_sample(list_of_experiments, xi , M ,  direct_fun = False , A = None):
    # pred_list = []
    index = torch.randint(0,len(list_of_experiments),(1,M))
    pred = []
    for m in index[0]:
        experiment = list_of_experiments[m]
        if direct_fun :
            pred.append(experiment(None, xi[:,None], A))
        else:
            pred.append(experiment.func(None, xi[:,None],A))
    pred = torch.stack(pred).squeeze()
    print(pred)
    pred = (pred.var(0).detach()).numpy()
    # pred_list.append(pred)
    return pred



def compute_d_statistics(list_of_experiments, x_test , M ,  direct_fun = False , A = None, number_of_draws = 100, n_id = None): #number_iterations = 1,
    pred_list = []
    nsamples = len(x_test)
    nnodes = x_test[0].shape[0]
    for i in range(number_of_draws):
        sample_idx = torch.randint(0,nsamples,[1])[0]
        if n_id == None:
            node_idx = torch.randint(0,nnodes,[1])[0]
        else:
            node_idx = n_id
        xi = x_test[sample_idx]
        pred = compute_d_statistics_one_sample(list_of_experiments, xi, M , direct_fun, A )
        pred = torch.tensor(np.array(pred)).squeeze()[node_idx].detach().numpy() #.mean(0).detach().numpy()[node_idx]
        pred_list.append((pred))
    return pred_list

def get_acc_ratio_sample_vs_null(null_samples, testing_samples, alpha):
    p_vals = []
    for niter in range(len(testing_samples)):
        dx = testing_samples[niter]
        p_val = compute_pval(dx, np.array(null_samples))
        p_vals.append(p_val)
    accepted = acceptance_ratio(p_vals, alpha)
    return accepted

def compute_pval( x_dval , d_stat_values): 
    return np.sum( d_stat_values >= x_dval ) / len(d_stat_values)
    

def acceptance_ratio(p_vals, alpha):
    return sum(np.array( p_vals ) >= alpha )/len(p_vals) 

# def compute_critical_val( d_stat_values , alpha):
#     x = 0.0
#     d_stat_values =  np.array(d_stat_values)
#     delta = 1/(len(d_stat_values))
#     while True:
#         pval = compute_pval(x, d_stat_values)
#         if pval <= alpha:
#             return x
#         x += delta


def compute_critical_val( d_stat_values , alpha):
    x, y = ecdf(d_stat_values)
    y = 1-y
    y[y < alpha] = 10
    return x[np.argwhere(y==10)[0]][0]

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

        
        