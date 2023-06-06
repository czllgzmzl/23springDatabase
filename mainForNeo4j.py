import string

from py2neo import Node, Relationship,Graph,NodeMatcher,Subgraph
import numpy as np
import networkx as nx

test_graph = Graph(
    "http://localhost:7474",
    # username='neo4j',
    password="www.lgz.comA1",
)

test_graph.delete_all()
#
# node_1 = Node('A',size = 3)
# node_2 = Node('B',size = 3)
# node_3 = Node('C',size = 3)
#
# test_graph.create(node_1)
# test_graph.create(node_2)
# test_graph.create(node_3)
# print(node_1)

dataset_path=[
    'karate.gml',
    'football.gml',
    'polblogs.gml',
]


def input_G_gml(path):
    Graph = nx.read_gml('./dataset/'+path,label='id')
    g = {}
    for (u,v,duplicated) in Graph.edges:
        if duplicated == 1:
            continue
        if u in g:
            g[u].append(v)
        else:
            g[u]=[v]
        if not Graph.is_directed():
            if v in g:
                g[v].append(u)
            else:
                g[v]=[u]
        if v not in g:
            g[v] = []
    G = {}
    for g_key in g:
        G[g_key] = np.asarray(g[g_key])
    # print(len(G))
    print('Graph.is_directed',Graph.is_directed())
    print('len(Graph.edges)',len(Graph.edges))
    return G


for path in dataset_path:
    dataset = path[0:path.index('.')]
    pathID = dataset_path.index(path)
    G=input_G_gml(path)
    nodes = {}
    id2ComName = string.ascii_uppercase
    for lineID,line in enumerate(open('./result/' + dataset+"_communities.txt", 'r')):
        nums = [int(x) for x in line.split('\t')]
        for g in nums:
            nodes[g] = Node(str(pathID)+id2ComName[lineID],node_id=str(g),dataset=dataset)
            test_graph.create(nodes[g])
        # print(nums)
        for num in nums:
            nodes[num].update({'community_size': len(nums),'community_id':id2ComName[lineID]})
            test_graph.push(nodes[num])
    for lineID,line in enumerate(open('./result/' + dataset+"_communities.txt", 'r')):
        sub_nodes,sub_edges=[],[]
        nums = [int(x) for x in line.split('\t')]
        for u in nums:
            sub_nodes.append(nodes[u])
            for v in G[u]:
                # u2v=Relationship(nodes[u],'I'if v in nums else 'O',nodes[v],style_color=0x2870c2if v in nums else 0xdddddd)
                u2v=Relationship(nodes[u],'I'if v in nums else 'O',nodes[v],style_color='#2870c2'if v in nums else '#dddddd')
                sub_edges.append(u2v)
        test_graph.create(Subgraph(nodes=sub_nodes,relationships=sub_edges))
