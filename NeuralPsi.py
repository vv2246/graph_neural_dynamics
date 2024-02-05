import torch
import torch.nn as nn
import networkx as nx
import numpy as np

class ODEFunc(nn.Module):

    def __init__(self, A, d, h, h2, h3, bias = True, Q_factorized = True):
        super(ODEFunc, self).__init__()

        self.A = A
        # self.d = d
        # self.n = A.shape[0]
        self.graph = nx.from_numpy_array(np.array(self.A.cpu()))
        I_scipy = nx.incidence_matrix(self.graph)
        #self.I = torch.sparse.FloatTensor(torch.LongTensor(I_scipy.nonzero()), torch.FloatTensor(I_scipy.data), torch.Size( I_scipy.shape))
        self.I = I_scipy
        self.Q_factorized = Q_factorized
        ###################################
        # Self interaction term
        ###################################

        self.g = nn.Sequential(
            nn.Linear(d, h, bias = bias),
            nn.Tanh(),
            nn.Linear(h, h, bias = bias),
            nn.Tanh(),
            nn.Linear(h, h, bias = bias),
            nn.Tanh(),
            nn.Linear(h, d, bias = bias),
        )

        ###################################
        # Interaction term
        ###################################
        if Q_factorized :
            self.nn_fun1 = nn.Sequential(
                nn.Linear(d, h2, bias = bias),
                nn.Tanh(),
                nn.Linear(h2, h2, bias = bias),
                nn.Tanh(),
                nn.Linear(h2, h2, bias = bias),
                nn.Tanh(),
                nn.Linear(h2, h2, bias = bias),
            )
            self.nn_fun2 = nn.Sequential(
                nn.Linear(d, h2, bias = bias),
                nn.Tanh(),
                nn.Linear(h2, h2, bias = bias),
                nn.Tanh(),
                nn.Linear(h2, h2, bias = bias),
                nn.Tanh(),
                nn.Linear(h2, h2, bias = bias),
            )
            # self.invariant_pooling_layer = lambda x: torch.sum(x, dim=1)[:, None]
            self.agg_input_min = 0.0
            self.agg_input_max = 0.0
            self.agg = nn.Sequential(
                nn.Linear(h2, d, bias = bias),
                # nn.Tanh(),
                # nn.Linear(h3, d, bias=bias),
            )
        else:
            self.Q = nn.Sequential(
                nn.Linear(2, h2, bias = bias),
                nn.Tanh(),
                nn.Linear(h2, h2, bias = bias),
                nn.Tanh(),
                nn.Linear(h2, h3, bias = bias)
                )#.to(device)

            self.agg = nn.Sequential(
                nn.Linear(h3, 1, bias = bias),
                # nn.Tanh(), # did this on nov 28
                # nn.Linear(h4, 1, bias=bias),
            )

            
    def forward(self, t, x, adj = None,  self_interaction = True, nbr_interaction = True ):
        if self.Q_factorized:
            return self.forward_factorized(t, x, adj, self_interaction, nbr_interaction)
        else:
            return self.forward_incidence(t, x, adj, self_interaction, nbr_interaction)
    
    def forward_incidence(self, t, x, adj = None,  self_interaction = True, nbr_interaction = True):
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
    
    def forward_factorized(self, t, x, adj = None, self_interaction = True, nbr_interaction = True):
        out = self.g(x)
        y1 = self.nn_fun1(x)
        y2 = self.nn_fun2(x)
        y1_tmp = torch.transpose(y1 ,0 ,2)
        y2_tmp = torch.transpose(y2 ,0 ,2)
        # print(y2_tmp.shape)
        xx = torch.bmm(torch.transpose(y1_tmp, 1 ,2) ,y2_tmp)
        # print(xx.shape)
        if adj == None:
            nn_Q_out = self.A * (xx)
        else:
            nn_Q_out = adj * xx
            
        # print(nn_Q_out.sum(1))
        nn_Q_flatten_input = torch.transpose( torch.unsqueeze(nn_Q_out.sum(1) , 1) , 0, 2).float()
        # print(nn_Q_flatten_input.shape)
        # print(nn_Q_flatten_input)
        agg_out = self.agg(nn_Q_flatten_input)
        # print(agg_out.shape, )
        # print(out, agg_out)
        # self.agg_input_min = min(self.agg_input_min, nn_Q_flatten_input.min().item())
        # self.agg_input_max = max(self.agg_input_max, nn_Q_flatten_input.max().item())
        res = 0
        if self_interaction == True:
            res += out
        if nbr_interaction == True : 
            res += agg_out 
        return res
    
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
    
    # d, h, h2, h3, h4 = 90, 40, 60, 50, 30
    # n = 5
    # x = torch.randn([n , 1 , d], requires_grad= True) # n x 1 x d    
    # g = nx.erdos_renyi_graph(n , 0.5)
    # A = torch.tensor(nx.to_numpy_array(g))
    # func = ODEFunc(A, d, h, h2, h3, h4, Q_factorized = True)
    # func( 0 , x, A) 
    
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
    
    
    
    
    
    
    
    