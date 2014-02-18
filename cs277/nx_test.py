__author__ = 'anbangx'

import networkx as nx

# networkx graph
G=nx.Graph()
# ad edges with red color
G.add_edge(1,2,color='red')
G.add_edge(2,3,color='red')
# add nodes 3 and 4
G.add_node(3)
G.add_node(4)

# convert to a graphviz agraph
A=nx.to_agraph(G)

# write to dot file
A.write('k5_attributes.dot')

# convert back to networkx Graph with attributes on edges and
# default attributes as dictionary data
X=nx.from_agraph(A)
print("edges")
print(X.edges(data=True))
print("default graph attributes")
print(X.graph)
print("node node attributes")
print(X.node)