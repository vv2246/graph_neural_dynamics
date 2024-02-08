import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# Replace 'your_file.gml' with the path to your GML file
gml_file = 'graphs/celegans_directed_wcc.gml'

# Read the GML file
G = nx.read_gml(gml_file)

G = nx.to_undirected(G)
# df = pd.read_csv("nodes_pos.csv", index_col= False)

# xpos = dict(zip(df['label'], df['xpos']))
# ypos = dict(zip(df['label'], df['ypos']))


# # Extract node positions from the GML file attributes
# # Assuming each node has 'x' and 'y' attributes for coordinates
pos = {node:(data['xpos'], data['ypos']) for node, data in G.nodes(data=True)}

# Node customization
node_colors = ["blue" for _ in G.nodes()]  # All nodes in bright blue
node_colors = []
edgecolors = []
for _ in G.nodes():
    if _ == "0" or _ == "137":
        node_colors.append("firebrick")
        edgecolors.append("darkred")
    else:
        node_colors.append("darkorange")
        edgecolors.append("chocolate")
node_sizes = [40 for _ in G.nodes()]  # Smaller size for all nodes

# Customize specific nodes
specific_nodes = ['0', '137']  # Nodes to be highlighted
# specific_node_sizes = [20 for _ in specific_nodes]  # Larger size for specific nodes
# specific_node_colors = ["red" for _ in specific_nodes]  # Same color for consistency


fig, ax = plt.subplots()
# Update sizes and colors for specific nodes
for node in specific_nodes:
    if node in G:
        index = list(G.nodes()).index(node)
        node_sizes[index] = 40  # Update size for specific nodes

min_alpha, max_alpha = 0.1, 1.0
weights = [G[u][v]['weight'] for u, v in G.edges()]
min_weight, max_weight = min(weights), max(weights)
alpha_range = max_alpha - min_alpha

# Draw the graph without edges first
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors=edgecolors,ax = ax)
# nx.draw_networkx_labels(G, pos, font_size=8)

for u, v, data in G.edges(data=True):
    # Normalize the weight of the edge
    norm_weight = (data['weight'] - min_weight) / (max_weight - min_weight)
    alpha = min_alpha + (norm_weight * alpha_range)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1, edge_color=f"royalblue", alpha = alpha, ax = ax)

# Manually adjust label positions to be at the corner
label_offset = 0.3  # Adjust as needed for your visualization
label_pos = {node: (pos[node][0] + label_offset, pos[node][1] + label_offset) for node in specific_nodes if node in G}


# Remove borders
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])

plt.axis('equal')
plt.tight_layout()
plt.savefig("figures/celegans_graph.pdf")
plt.savefig("figures/celegans_graph.png")
plt.show()
# # Draw labels for specific nodes
# labels = {node:node for node in specific_nodes if node in G}
# nx.draw_networkx_labels(G, pos, labels, font_size=12)
