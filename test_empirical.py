
from utilities import load_results
from dynamics import Dynamics
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import from_networkx
import networkx as nx
from torchdiffeq import odeint
plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (6,6)
def plot_y_ytest(t,y,y_test, label, idx = 50 ):
    print(abs(y[:,idx,:]-y_test[:,idx,:]).mean())
    for i in range(2):
        plt.plot(t,y[:,idx,i], color= "blue", label = "True")
        plt.plot(t,y_test[:,idx,i],linestyle="--", color= "navy", label = "Predicted")
        plt.legend(loc =1, facecolor='white', framealpha = 1)
        plt.ylabel(f"$x_{i}(t)$")
        plt.xlabel('t')
        plt.title(label)
        plt.show()
    
    plt.plot(y_test[:,idx,0],y_test[:,idx,1] ,color= "blue", label = "True")
    plt.plot(y[:,idx,0],y[:,idx,1], linestyle = "--", color= "navy", label = "Predicted")
    plt.legend(loc =1, facecolor='white', framealpha = 1)
    plt.ylabel(f"$x_1(t)$")
    plt.xlabel(f"$x_0(t)$")
    plt.title(label)
    plt.show()

# graph
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
A, params, func, x_train,y_train  = load_results("results/FHN_celegans_directed_wcc_multiple_nn_True_0", device)

dt_base = 10**-2
T = 50
t = torch.linspace(0,T, int(T/dt_base))

# L =  A/  torch.diag(A.sum(0))
# L[torch.isnan(L)] = 0.

dyn =  Dynamics(A=A,  model = params.dynamics_name)
x0_test = torch.rand([params.size,params.d])
y = odeint( dyn, x0_test, t, method="dopri5")
# y_test = odeint(lambda y, t: func(y, t), x0_test[:,None], t, method="dopri5").squeeze().detach()
y_test = odeint(func, x0_test[:,None], t, method="dopri5").squeeze().detach()


plot_y_ytest(t,y,y_test, "Train graph",idx = 4)



dt_base = 10**-2
T = 50
g_test = nx.read_gml(f"graphs/barabasi_albert_N_100_m_3.gml")
A_test = torch.FloatTensor(np.array(nx.adjacency_matrix(g_test).todense()))
L_test =  A_test/  (A_test.sum(0))
L_test[torch.isnan(L_test)] = 0
dyn_test =  Dynamics(A=A_test,   model = params.dynamics_name)
x0_test = torch.rand([A_test.shape[0],params.d])

y = odeint( dyn_test, x0_test, t, method="dopri5")
y_test = odeint(lambda y, t: func(y, t, L_test), x0_test[:,None], t, method="dopri5").squeeze().detach()


plot_y_ytest(t,y,y_test, label = "New graph")
