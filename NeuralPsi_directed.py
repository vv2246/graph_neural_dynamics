import torch
import torch.nn as nn
import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import networkx as nx
import numpy as np

class ODEFunc(nn.Module):

    def __init__(self, A, d, h, h2, h3, bias=True, Q_factorized=True):
        super(ODEFunc, self).__init__()

        self.A = A
        self.graph = nx.from_numpy_array(np.array(self.A.cpu()), create_using=nx.DiGraph)  # Ensure graph is directed
        I_scipy = nx.incidence_matrix(self.graph, oriented=True)  # Get oriented incidence matrix for directed graph
        self.I = I_scipy
        self.Q_factorized = Q_factorized

        ###################################
        # Self interaction term
        ###################################
        self.g = nn.Sequential(
            nn.Linear(d, h, bias=bias),
            nn.Tanh(),
            nn.Linear(h, h, bias=bias),
            nn.Tanh(),
            nn.Linear(h, d, bias=bias),
        )

        ###################################
        # Interaction term for directed graphs
        ###################################
        if Q_factorized:
            self.nn_fun_source = nn.Sequential(
                nn.Linear(1, h2, bias=bias),
                nn.Tanh(),
                nn.Linear(h2, h2, bias=bias),
                nn.Tanh(),
                nn.Linear(h2, h2, bias=bias),
            )
            self.nn_fun_target = nn.Sequential(
                nn.Linear(1, h2, bias=bias),
                nn.Tanh(),
                nn.Linear(h2, h2, bias=bias),
            )
            self.agg = nn.Sequential(
                nn.Linear(h2, d, bias=bias),
            )
        else:
            # If needed, define a non-factorized approach for directed graphs
            pass
            out = self.g(x)
            if adj == None:
                I = self.I
                graph = self.graph
            else:
                graph = nx.from_numpy_array(np.array(adj))
                I_scipy = nx.incidence_matrix(graph).todense()
                #I = torch.sparse.FloatTensor(torch.LongTensor(I_scipy.nonzero()), torch.FloatTensor(I_scipy.data), torch.Size( I_scipy.shape))
                I = torch.FloatTensor(I_scipy)

            node_1_list = [u for (u,v) in graph.edges()] 
            node_2_list = [v for (u,v) in graph.edges()]
            x_ij = torch.cat((x[node_1_list], x[node_2_list] ), 1) # e x 2 x d
            res = self.Q(torch.transpose(x_ij,1 ,2 )) # e x d x h3
            res = torch.transpose(res, 0, 2) # h3 x d x e
            I = torch.transpose(I, 0, 1) # e x n
            agg_out = torch.matmul(res,I)
            agg_out = self.agg(torch.transpose(agg_out, 0, 2))
            agg_out = torch.transpose(agg_out, 1, 2)
            res = 0
            if self_interaction == True:
                res += out
            if nbr_interaction == True : 
                res += agg_out 
            return res
    
    def forward(self, t, x, adj=None, self_interaction=True, nbr_interaction=True):
        out = self.g(x)
    
        if adj is None:
            adj = self.A
        else:
            adj = torch.tensor(adj, dtype=torch.float32)
    
        # Ensure adj is a tensor and add a batch dimension if necessary
        # adj = adj.unsqueeze(2)  # Add batch dimension to adjacency matrix
        # print(adj.shape)
    
        # Compute transformations for source and target node features
        source_features = self.nn_fun_source(x.transpose(1,2)).unsqueeze(1)
        target_features = self.nn_fun_target(x.transpose(1,2)).unsqueeze(0)
        
        xx = source_features * target_features
        # xx = xx.tra
        # print(xx.shape)
        
    
        # Perform batch matrix multiplication and element-wise multiplication
        directed_interaction =   xx * adj.unsqueeze(-1).unsqueeze(-1) #* target_features
        directed_interaction = directed_interaction.sum(0).sum(2).unsqueeze(1)
        # print(directed_interaction.shape)
        # print(directed_interaction)
    
        # # Sum over source nodes for each target, remove batch dimension, and apply final transformation
        # nn_Q_out = directed_interaction.sum(0)[:,None,:]#.squeeze(0)  # Squeeze to remove the batch dimension added earlier
        # # print(nn_Q_out)
        # agg_out = self.agg(nn_Q_out)
        # # print(agg_out)
        # res = out
        # if nbr_interaction:
        #     res += agg_out
        
        # print(out.shape,directed_interaction.shape)
        return out  + directed_interaction
        
class ODEFuncFull(nn.Module):
    def __init__(self, n, d):
        super(ODEFuncFull, self).__init__()

        self.g = nn.Sequential(
            nn.Linear(n, d),
            nn.Tanh(),
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Linear(d, n),
        )

    def forward(self, t, x):
        out = self.g(torch.transpose(x, 0,1))
        return torch.transpose(out,0,1)
    
    
if __name__ == "__main__":
    
    d, h, h2, h3, h4 = 90, 40, 60, 50, 30
    n = 3
    x = torch.randn([n , 1 , d], requires_grad= True) # n x 1 x d    
    g = nx.erdos_renyi_graph(n ,1)
    g = nx.DiGraph()
    g.add_edge(0,1)
    g.add_edge(2,1)
    A = torch.tensor(nx.to_numpy_array(g))
    func = ODEFunc(A, d, h, h2, h3, h4, Q_factorized = True)
    func( 0 , x, A) 
    
    
    
    
    
    
    