import networkx as nx
import matplotlib.pyplot as plt
data = './dataset/football.gml'
Graph=nx.read_gml(data,label='id')
nx.draw(Graph, with_labels=True)
print(Graph.nodes)
print(Graph.edges)
plt.show()
