import csv
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import networkx as nx
import numpy as np
# get the zipped file from link below
address = "https://networks.skewed.de/net/celegansneural/files/celegansneural.csv.zip"
resp = urlopen(address)
zipfile = ZipFile(BytesIO(resp.read()))

df = pd.read_csv(zipfile.open('nodes.csv'), index_col= False)
# output CSV has the three columns: label, position on x, position on y
with open("nodes_pos.csv", "w") as csvfile:
 csv_writer = csv.writer(csvfile)
 csv_writer.writerow(['label', 'xpos', 'ypos'])
 for row in np.array(df):
     csv_writer.writerow([row[1], float(row[2].split(",")[0][7:]), float(row[2].split(",")[1][:-2])])


df = pd.read_csv("nodes_pos.csv", index_col= False)

xpos = dict(zip(df['label'], df['xpos']))
ypos = dict(zip(df['label'], df['ypos']))

# # import the list of edges from edges.csv
df = pd.read_csv(zipfile.open('edges.csv'))
# # save a list of edges in the form: (source, target, weight)
edges_list = list(zip(df[df.columns[0]], df[df.columns[1]], df[df.columns[2]]))# recover an undirected weighted graph from edges list
G = nx.DiGraph()
G.add_weighted_edges_from(edges_list)
# # assign positions in space as node attributes
nx.set_node_attributes(G, xpos, 'xpos')
nx.set_node_attributes(G, ypos, 'ypos')
# # save the deriving graph in .graphml format

list_of_subgraphs = list(nx.weakly_connected_components(G))
list_of_digraphs = []
for subgraph in list_of_subgraphs:
    list_of_digraphs.append(nx.subgraph(G, subgraph))
max_wcc = max(nx.weakly_connected_components(G), key=len)
max_wcc = nx.subgraph(G, max_wcc)

nx.write_graphml(max_wcc, "graphs/celegans_directed_wcc.graphml")
nx.write_gml(max_wcc, "graphs/celegans_directed_wcc.gml")