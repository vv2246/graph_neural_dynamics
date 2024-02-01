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
from torchdiffeq import odeint
from utilities import save_results, ModelParameters, GCN, GCN_single
from torch_geometric.utils import from_networkx
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from utilities import save_results
    
if __name__ == "__main__":
    
    warnings.filterwarnings('ignore')
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    multiple_nn = True
    if multiple_nn:
        M_tot = 1
        bootstrap_fraction = 0.9
    else:
        M_tot = 1
        bootstrap_fraction  = 1
        
    dynamics_name = "FHN"
    network_name = "celegans_directed_wcc"
    # network_name = "erdos_renyi_N_10_p_0.1"
    model_name = "NeuralPsi"
    results_root = f"results/{dynamics_name}_{network_name}_multiple_nn_{multiple_nn}"
    g = nx.read_gml(f"graphs/{network_name}.gml")
    
    # d= nx. DiGraph()
    # for i,j in g.edges():
    #     if random.random()<0.5:
    #         d.add_edge(i,j)
    #     else:
    #         d.add_edge(j,i)
            
        
    
    A = torch.FloatTensor(np.array(nx.adjacency_matrix(g).todense()))
    L = A / A.sum(0)
    L[torch.isnan(L)] = 0
    
    regularizer_lambda = 1.0 # regularizer that minimizes variance in the loss across nodes
    params = ModelParameters(
        dynamics_name = dynamics_name,
        model_name =model_name,
        train_distr= torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1])),
        epochs = 10000,
        train_samples = 2000,
        size = A.shape[0],
        lr = 0.002,  # learning rate for training
        weight_decay = 0.00, # weight decay for training
        h = 30, #self interaction embedding dim
        h2 = 30 ,#nbr interaction embedding dim
        h3 = 0,
        d = 2,
        Q_factorized = True,
        bias = True)
    
    params.train_distr.sample([params.size]).to(device)
    if params.dynamics_name == "MAK":
        params.B = 0.1
        params.R = 1
        params.F = 0.5
        params.b =1
        dyn = Dynamics(A=A, B= params.B, R= params.R, F=params.F, b=params.b,  model = params.dynamics_name)
    if params.dynamics_name == "SIS":
        params.B = 4
        params.R = 5
        dyn = Dynamics(A=A, B= params.B, R =params.R,  model = params.dynamics_name)
    if params.dynamics_name == "Diffusion":
        params.B = 0.5
        dyn = Dynamics(A=A, B= params.B,   model = params.dynamics_name)
    if params.dynamics_name == "PD":
        params.B = 2
        params.R = 0.3
        params.a = 1.5
        params.b = 3
        dyn = Dynamics(A=A, B= params.B, R= params.R, a= params.a, b= params.b,   model = params.dynamics_name)
    if params.dynamics_name == "MM":
        params.H = 3
        params.B = 4
        params.R = 0.5
        dyn = Dynamics(A=A, B= params.B, R = params.R, H=params.H,   model = params.dynamics_name)
    else:# params.dynamics_name == "HR":
        dyn = Dynamics(A=A, model = params.dynamics_name)
        
        
    ######
    # Generate data
    ######
    dt_base = 10**-2
    T = 100
    t = torch.linspace(0,T, int(T/dt_base))
    scale_obs = 0.0
    y_nonoise = []
    y_train = []
    x_train = []
    t_train = []
    niter = 5#int( params.train_samples / t.shape[0] )
    
    for i in range(niter):
        
        x0 = torch.rand([params.size,params.d])
            
        y = odeint( dyn, x0, t, method="dopri5")#[20000:]
        plt.plot(y[:,0,0])
        plt.show()
        try:
            m = torch.distributions.normal.Normal(loc = 0, scale = scale_obs)
            noise = m.sample(y.shape)
        except:
            noise = 0
            print("var is zero")
        y_noise = y + noise
        y_train_i = (y_noise[1:,:,:] - y_noise[:-1,:,:])/dt_base # x_t+1 -x_t / dt
        x_train_i = y_noise[:-1,:,:] 
        for i in range(len(y_train_i)):
            y_train.append(y_train_i[i])
            x_train.append(x_train_i[i])
                
    ntrain = len(y_train)
    n_bootstrap = int(bootstrap_fraction * ntrain)
    print("integrated")
    #########################
    # do bootstrap of DATA
    #########################
    x_train_tot , y_train_tot = [],[]
    for m in range(M_tot):
        index = torch.randint(0,ntrain,(1,n_bootstrap))
        x_train_tot.append([x_train[i] for i in index[0]])
        y_train_tot.append([y_train[i] for i in index[0]])
    
    
    if params.model_name == "NeuralPsi":
        models = [ODEFunc(A = L, d = params.d, h=params.h, h2=params.h2,h3 = params.h3,
                            bias = True,  Q_factorized= True).to(device) for i in range(M_tot)] 
        
    
    #####
    # Training
    #####
    nsample = 200
    for exp_iter in range(M_tot):
        print(exp_iter)
        func = models[exp_iter]
        optimizer = torch.optim.Adam(func.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience= 50, cooldown=10)
        x_train_exp = x_train_tot[exp_iter]
        y_train_exp = y_train_tot[exp_iter]
        for itr in range(params.epochs + 1 ):
            optimizer.zero_grad()
            # take bootstrapped sample 
            index = torch.randint(0,n_bootstrap,(1,nsample))
            pred_y = [func(0, x_train_exp[i][:,None,:]) for i in index[0]]
            y_train_batch = [y_train_exp[i] for i in index[0]]
            
            # compute loss
            v1= torch.stack(pred_y).squeeze()
            v2 = torch.stack(y_train_batch)
            l_idx =torch.abs(v1-v2)
            
            # weigh wrt to variance in the loss
            node_var = l_idx.var(1)
            variance_reg = (node_var.mean() )
            loss = variance_reg * regularizer_lambda  + l_idx.mean()
            
            # # # compute val loss
            # if itr > 500 :
            #     pred_y_val = [func(0, x_train[i][:,None,:]) for i in  range(len(x_train))]
            #     y_val = [y_train[i] for i in range(len(y_train))]
            #     y_val = torch.stack(y_val)
            #     pred_y_val = torch.stack(pred_y_val).squeeze()
            #     loss_tot_val = torch.abs(pred_y_val - y_val).mean()
            #     prev_lr = optimizer.param_groups[0]['lr']
            #     scheduler.step(loss_tot_val)
            #     if prev_lr != optimizer.param_groups[0]['lr']:
            #         print("learning rate scheduler update: ", {optimizer.param_groups[0]['lr']})
            
            loss.backward()
            optimizer.step()
            if itr % 100 == 0:
                with torch.no_grad():
                    print(itr, loss)
    
        save_results(model = func, folder_name = results_root+f"_{exp_iter}", adj = L, training_parameters = params,x_train = x_train, y_train = y_train)

    
