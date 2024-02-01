import networkx as nx 
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import warnings
from NeuralPsi import ODEFunc
import random
warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

N = 100
p = 0.1

connected = False
while connected == False:
    g = nx.erdos_renyi_graph(N, p)
    if nx.is_connected(g) == True:
        connected = True 
nx.write_gml(g, f"graphs/erdos_renyi_N_{N}_p_{p}.gml")

g = nx.watts_strogatz_graph(N, 4, 0.1 )
nx.write_gml(g, f"graphs/watts_strogatz_N_{N}.gml")

m = 3
g = nx.barabasi_albert_graph(N , m)
nx.write_gml(g, f"graphs/barabasi_albert_N_{N}_m_{m}.gml")



N = 10
p = 0.1

connected = False
while connected == False:
    g = nx.erdos_renyi_graph(N, p)
    if nx.is_connected(g) == True:
        connected = True 
nx.write_gml(g, f"graphs/erdos_renyi_N_{N}_p_{p}.gml")