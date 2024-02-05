from utilities import  load_results, compute_d_statistics_one_sample,compute_critical_val,  compute_d_statistics, compute_pval ,set_seeds,acceptance_ratio, get_acc_ratio_sample_vs_null
from dynamics import Dynamics
# from test_significance import test_significance_functions
import networkx as nx
import matplotlib.pyplot as plt
import warnings
# from small_example_repurified import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
import torch 
from torchdiffeq import odeint
import numpy as np
import os



def generate_statistics(A_test , delta_test, true_dyn, time_array,  d_stat_full_sample,
                        neural_net_list, M , alpha_sig = 0.05, number_of_iterations =10 ):
    res = []
    for niter in range(number_of_iterations) :
        x0 = m.sample([A_test.shape[0]]) + delta_test
        y = odeint( true_dyn, x0, time_array, method= 'dopri5' ).squeeze().t()
        # sol = pooled_integral(neural_net_list, x0[:,None], time_array, A_test)
        idx = np.random.randint(M)
        sol = odeint(lambda y, t: neural_net_list[idx](y, t, A_test), x0[:,None], time_array, method="dopri5").squeeze().detach()
        loss = [] 
        x_pred  = [] 
        for k in range(y.shape[1]):
            x_pred.append(sol[k,:][:,None]) 
            # loss.append(torch.stack( [ abs(dyn(0, y[:,k][:,None]).squeeze() - neural_net_list[idx](0, y[:,k][:,None,None], A_test ).squeeze()) ])#for func in neural_net_list ]) )
            loss.append(abs(dyn(0, y[:,k][:,None]).squeeze() - neural_net_list[idx](0, y[:,k][:,None,None], A_test ).squeeze()))#for func in neural_net_list ]) )
        
        loss = float(torch.stack(loss).mean().detach().numpy())
        sigpred = compute_d_statistics(list_of_experiments = neural_net_list, 
                                       x_test = x_pred, M = M , direct_fun = True, 
                                       A = A_test, number_of_draws = 100)
        accepted = get_acc_ratio_sample_vs_null(null_samples = d_stat_full_sample, 
                                       testing_samples = sigpred, alpha = alpha_sig )
        res.append([true_dyn.model, delta_test, loss, accepted])
    return res 



    

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    # load neural networks
    
    adjacencies = []
    x_train_full = []
    y_train_full = []
    loss_full = []
    nn_list = []
    for i in range(20):
        A, params, func, x_train,y_train  = load_results(f"results/FHN_celegans_directed_wcc_multiple_nn_True_{i}", device)
        # adjacencies.append(A)
        nn_list.append(func)
        x_train_full.append(x_train)
        y_train_full.append(y_train)
        # break
    
    # generate statistics from training
    d_stat_full_sample = compute_d_statistics(list_of_experiments = nn_list,  x_test = x_train , M = 20, 
                                              direct_fun = True, number_of_draws = 100, A = A)
    
    # generate statistics in a new setting
    dt_base = 10**-2
    T = 50
    t = torch.linspace(0,T, int(T/dt_base))
    g_test = nx.read_gml(f"graphs/barabasi_albert_N_100_m_3.gml")
    A_test = torch.FloatTensor(np.array(nx.adjacency_matrix(g_test).todense()))
    L_test =  A_test/  (A_test.sum(0))
    L_test[torch.isnan(L_test)] = 0
    dyn_test =  Dynamics(A=A_test,   model = params.dynamics_name)
    x0_test = torch.rand([A_test.shape[0],params.d])
    
    # compute all solutions 
    solutions = []
    for func in nn_list:
        sol = odeint(lambda y, t: func(y, t, L_test), x0_test[:,None], t, method="dopri5").squeeze().detach()
        solutions.append(sol)
    
    
    plt.plot(t, torch.stack(solutions).var(0).mean(1))
    plt.ylabel("$E[Var[\\mathbf{x}(t)]_m]_{i}$")
    plt.xlabel("$t$")
    plt.show()
    
    for t_idx in range(t.shape[0]):
        xi = sol[t_idx]
        d_stat = compute_d_statistics_one_sample(nn_list, xi, 20,direct_fun = True, A = A_test)
        break
        
    sigpred = compute_d_statistics(list_of_experiments = nn_list, 
                                    x_test = sol, 
                                    M = 20 , direct_fun = True, A = A_test, number_of_draws = 100)
    
    
    
    
    dyn_test =  Dynamics(A=A_test,   model = params.dynamics_name)
    y_pred = odeint( dyn_test, x0_test, t, method="dopri5")
    

