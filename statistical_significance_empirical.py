from utilities import  load_results, compute_d_statistics_one_sample,compute_critical_val,  compute_d_statistics, compute_pval ,set_seeds,acceptance_ratio, get_acc_ratio_sample_vs_null
from dynamics import Dynamics
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
import torch 
from torchdiffeq import odeint
import numpy as np
import os


# def compute_d_statistics_one_sample(list_of_experiments, xi , M ,  direct_fun = False , A = None):
#     # pred_list = []
#     index = torch.randint(0,len(list_of_experiments),(1,M))
#     # print(index)
#     pred = []
#     for m in index[0]:
#         experiment = list_of_experiments[m]
#         if direct_fun :
#             pred.append(experiment(None, xi[:,None], A))
#         else:
#             pred.append(experiment.func(None, xi[:,None],A))
#     pred = torch.stack(pred).squeeze()
#     # print(pred.shape)
#     pred = (pred.var(0).detach()).numpy()
#     # print(pred)
#     # pred_list.append(pred)
#     return pred



# def compute_d_statistics(list_of_experiments, x_test , M ,  direct_fun = False , A = None, number_of_draws = 100, n_id = None): #number_iterations = 1,
#     pred_list = []
#     nsamples = len(x_test)
#     nnodes = x_test[0].shape[0]
#     for i in range(number_of_draws):
#         sample_idx = torch.randint(0,nsamples,[1])[0]
#         if n_id == None:
#             node_idx = torch.randint(0,nnodes,[1])[0]
#         else:
#             node_idx = n_id
#         xi = x_test[sample_idx]
#         # print(xi)
#         pred = compute_d_statistics_one_sample(list_of_experiments, xi, M , direct_fun, A )
#         # print(pred)
#         pred = torch.tensor(np.array(pred)).squeeze()[node_idx].detach().numpy() #.mean(0).detach().numpy()[node_idx]
#         # print(pred)
#         pred_list.append((pred))
#     return pred_list
    

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    generate_solutions = False
    # load neural networks
    set_seeds()
    
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
    
    network_name = "celegans_directed_wcc"
    # network_name = "simple_directed"
    # model_name = "NeuralPsi"
    # results_root = f"results/{dynamics_name}_{network_name}_multiple_nn_{multiple_nn}"
    g = nx.read_gml(f"graphs/{network_name}.gml")
    
    # generate statistics from training
    d_stat_full_sample = np.stack(compute_d_statistics(list_of_experiments = nn_list,  x_test = x_train , M = 5, 
                                              direct_fun = True, number_of_draws = 1000, A = A))
    
    d_stat_full1 =  d_stat_full_sample[:,0]
    d_stat_full2 =  d_stat_full_sample[:,1]
    dt_base = 10**-2
    T = 100
    t = torch.linspace(0,T, int(T/dt_base))
    A_test = A#.copy()
    L_test =  A_test/  (A_test.sum(0))
    L_test[torch.isnan(L_test)] = 0
    
    x0_test = torch.rand([A_test.shape[0],params.d])
    sol = odeint(lambda y, t: nn_list[0](y, t, L_test), x0_test[:,None], t, method="dopri5").squeeze().detach()
    

    dyn_test =  Dynamics(A=A_test,   model = params.dynamics_name)
    y_pred = odeint( dyn_test, x0_test, t, method="dopri5")

    i = 0
    fig,ax = plt.subplots(nrows = 2, figsize= (15,6),sharex= True)
    ax[0].plot(t,sol[:,i,0], label="True", color = "darkorange", linestyle = "-", lw = 2,alpha =1)
    ax[1].plot(t,sol[:,i,1], label="True", color = "darkorange", linestyle = "-", lw = 2,alpha = 1)
    # plt.plot(t,solutions[1,:,0,0], label="True", color = "orange", linestyle = "-", lw = 0.5,alpha = 0.5)
    ax[0].plot(t, y_pred[:,0,0], color = "blue", lw = 2, linestyle = "--")
    ax[1].plot(t, y_pred[:,0,1], color = "blue", lw = 2, linestyle = "--")
    ax[0].set_title(f"$i={i}$")
    # plt.show()
    
    ax[1].set_xlabel("$t$")
    ax[0].set_ylabel("$x_{i,0}(t)$")
    ax[1].set_ylabel("$x_{i,1}(t)$")
    ax[0].set_ylim(-2.5,2.5)
    ax[1].set_ylim(-2.5,2.5)
    ax[1].set_xlim(0,100)
    plt.savefig(f"figures/node_{i}_timeseries.pdf")
    plt.show()
    
    fig,ax = plt.subplots(nrows = 1, figsize= (6,6),sharex= True)
    ax.plot(sol[:,i,0], sol[:,0,1],label="True", color = "darkorange", linestyle = "-", lw = 2,alpha =1)
    ax.plot(y_pred[:,i,0], y_pred[:,0,1],color = "blue", lw = 2, linestyle = "--")
    ax.set_xlabel("$x_{i,0}(t)$")
    ax.set_ylabel("$x_{i,1}(t)$")
    ax.set_ylim(-1.5,2)
    ax.set_xlim(-2,1.5)
    plt.tight_layout()
    plt.savefig(f"figures/node_{i}_phase_portrait.pdf")
    plt.show()
    
    d_stat_full_sample = np.stack(compute_d_statistics(list_of_experiments = nn_list,  x_test = x_train , M = 5, 
                                              direct_fun = True, number_of_draws = 1000, A = A, n_id= i ))
    
    d_stat_test = np.stack(compute_d_statistics(list_of_experiments = nn_list,  x_test = sol , M = 5, 
                                              direct_fun = True, number_of_draws = 1000, A = A,n_id = i))
    d_stat_test1 = d_stat_test[:,0]
    d_stat_test2 = d_stat_test[:,1]
    
    fig,axs = plt.subplots(nrows = 2 , figsize= (6,8),sharey=True)
    ax , ax2 = axs
    sns.ecdfplot(d_stat_test1, complementary= True, color = "darkorange", linewidth=2,alpha= 1, ax = ax)
    sns.ecdfplot(d_stat_full1, complementary = True, color = "blue", linewidth=2,alpha= 1,ax = ax)
    ax.set_xlim(0,compute_critical_val(d_stat_full_sample[:,0] , alpha =0.05))
    accepted1 = get_acc_ratio_sample_vs_null(null_samples = d_stat_full1, testing_samples = d_stat_test1, alpha = 0.05)
    ax.set_title(f"$k=0$, {round(accepted1*100)}% accepted")#" accepted " f"{round(accepted1*100)}%")
    # ax.set_xticks([0, 0.002, 0.004])
    ax.set_xlabel("$d$")
    
    # fig,ax = plt.subplots(figsize= (6,6))
    sns.ecdfplot(d_stat_test2, complementary= True, color = "darkorange", linewidth=2,alpha= 1, ax = ax2)
    sns.ecdfplot(d_stat_full2, complementary = True, color = "blue", linewidth=2,alpha= 1,ax = ax2)
    ax2.set_xlim(0,compute_critical_val(d_stat_full2 , alpha =0.05))
    accepted2 = get_acc_ratio_sample_vs_null(null_samples = d_stat_full2, testing_samples = d_stat_test2, alpha = 0.05)
    ax2.set_title(f"$k=1$, {round(accepted2*100)}% accepted")# accepted " f"{round(accepted1*100)}%")
    ax2.set_xlabel("$d$")
    # ax2.set_xticks([0, 0.0001,0.0002])
    plt.tight_layout()
    plt.savefig(f"figures/d_statistic_node_{i}.pdf")
    plt.show()


    y_train_pred = torch.stack([ func(0,x_train[i][:,None,:]) for i in range(x_train.shape[0])]).squeeze()
    train_loss = abs(y_train - y_train_pred).mean(0).detach()
    
    fig,ax = plt.subplots()
    ax.hist(train_loss[:,1])
    
    test_loss = abs(sol - y_pred).mean(0)
    ax.hist(test_loss[:,1])
