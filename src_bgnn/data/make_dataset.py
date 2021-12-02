from src.data import utils as ut
from src.data import config
from src.features import build_features as bf
import pickle
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import random
from src_bgnn.data import config as cnf
import scipy.io as io
## generate training data

# graphsizelist = [100, 200, 400, 400, 500, 500, 800, 800, 1000, 1000, 2000, 2000, 3000, 4000]
# graphsizelist = [ 200, 300, 500, 500, 600, 600, 700, 700, 800, 800]
graphsizelist = [10700]

fileext = "\plc_"+ str(sum(graphsizelist))

# generate graph and labels (ranks) for synthetic plcluster networks
# Listgraph, Listlabel = ut.get_graphfeaturelabel_syn('plc', 'egr', graphsizelist)

## generate egr score for synthetic graphs
#
# md = ut.GenEgrData()
# Listgraph =[]
# Listlabel = []
#
# def get_bagraphegrscores(graphsizelist):
#
#     for gsize in graphsizelist :
#
#         # g = nx.generators.random_graphs.powerlaw_cluster_graph(gsize, 4, 0.05)
#         g = nx.barabasi_albert_graph(n=gsize, m=3)
#
#         nodelist = list(g.nodes)
#         # g2 = nx.scale_free_graph(300).to_undirected()
#         # egrorggraph = md.get_egr(g2)
#         egrdict = md.get_egrdict(g, nodelist)
#         # egrranks = rankdata(egrdict, method='dense')
#         # egrnorm = (egrranks - min(egrranks))/( max(egrranks) - min(egrranks))
#         # plt.hist(egrnorm)
#         Listgraph.append(g)
#         Listlabel.append(egrdict)
#         print(gsize)
#
#     return Listlabel, Listgraph
# Listlabel, Listgraph = get_bagraphegrscores(graphsizelist)
#
# def get_plgraphegrscores(graphsizelist):
#
#     for gsize in graphsizelist :
#
#         g = nx.scale_free_graph(gsize).to_undirected()
#         # g2 = nx.scale_free_graph(300).to_undirected()
#         # egrorggraph = md.get_egr(g2)
#         egrdict = md.get_egrdict(g)
#         # egrranks = rankdata(egrdict, method='dense')
#         # egrnorm = (egrranks - min(egrranks))/( max(egrranks) - min(egrranks))
#         # plt.hist(egrnorm)
#         Listgraph.append(g)
#         Listlabel.append(egrdict)
#         print(gsize)
#
#     return Listlabel, Listgraph
# Listlabel, Listgraph = get_plgraphegrscores(graphsizelist)
#
# def get_plcgraph_wsscores(graphsizelist):
#     Listgraph = []
#     Listlabel = []
#     for gsize in graphsizelist :
#
#         g = nx.generators.random_graphs.powerlaw_cluster_graph(gsize, 4, 0.05)
#         egrdict = md.get_wghtspectnode(g)
#         Listgraph.append(g)
#         Listlabel.append(egrdict)
#         print(gsize)
#
#     return Listlabel, Listgraph
# Listlabel, Listgraph = get_plcgraph_wsscores(graphsizelist)
#
# def get_randomgeograph_egrscores(graphsizelist):
#     Listgraph = []
#     Listlabel = []
#     for gsize in graphsizelist :
#
#         g = nx.random_geometric_graph(gsize, 0.125)
#
#         check=1
#         while check==1:
#             if nx.is_connected(g):
#                 check = 0
#             else:
#                 print("disconnected found")
#                 g = nx.random_geometric_graph(gsize, 0.125)
#
#         nodelist = list(g.nodes)
#         egrdict = md.get_egrdict(g, nodelist)
#         if np.var(egrdict) < 1e-10:
#             continue
#         Listgraph.append(g)
#         Listlabel.append(egrdict)
#         print(gsize)
#
#     return Listlabel, Listgraph
# Listlabel, Listgraph = get_randomgeograph_egrscores(graphsizelist)
#
# def get_wattstrogatzgraph_egrscores(graphsizelist):
#     Listgraph = []
#     Listlabel = []
#     for gsize in graphsizelist :
#
#         g = nx.connected_watts_strogatz_graph(n=gsize, k=6, p=0.1)
#
#         check=1
#         while check==1:
#             if nx.is_connected(g):
#                 check = 0
#             else:
#                 print("disconnected found")
#                 g = nx.random_geometric_graph(gsize, 0.125)
#
#         nodelist = list(g.nodes)
#         egrdict = md.get_egrdict(g, nodelist)
#         if np.var(egrdict) < 1e-10:
#             continue
#         Listgraph.append(g)
#         Listlabel.append(egrdict)
#         print(gsize)
#
#     return Listlabel, Listgraph
# Listlabel, Listgraph = get_wattstrogatzgraph_egrscores(graphsizelist)
#
# with open(config.datapath + 'processed'+ fileext + "_wsscore.pickle", 'wb') as b:
#     pickle.dump(Listlabel, b)
#
# with open(config.datapath + 'processed'+ fileext + "_listgraph.pickle", 'wb') as b:
#     pickle.dump(Listgraph, b)

## ================= load egr score of graph

with open(config.datapath + 'processed\\' + fileext+ "_egrscore.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

with open(config.datapath + 'processed\\' + fileext+ "_listgraph.pickle", 'rb') as b:
    Listgraph = pickle.load(b)

# combine graphs into one disjoint union graph
g = bf.combine_graphs(Listgraph)

## get ranks from Listlabels egr scores

Listranks=[]

for countlist in Listlabel:
    egrdict = rankdata(countlist, method='dense')
    egrranknorm = (egrdict - min(egrdict)) / (max(egrdict) - min(egrdict))
    Listranks.append(egrranknorm)

## generate target vector for syn graph

nodelist = np.arange(0, len(g.nodes))
targetdf = bf.getgraphtargetdf(Listranks, nodelist)
plt.figure(1)
plt.plot(targetdf['metric'])
plt.hist(targetdf.metric)
plt.close(1)

##  generate class label for classification

# targetdf.loc[targetdf.metric==0,'metric'] = 0.001
#
# category = pd.cut(targetdf.metric, bins=[0,0.8,0.95,1.0],labels=[0,1,2])
# targetdf['metric'] = category
# plt.hist(targetdf.metric)
# targetdf = pd.get_dummies(targetdf.metric)

### feature vector

for node_id, node_data in g.nodes(data=True):
    node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1, 1]
    # node_data["feature"] = [0.2, 0.033, 0.04, 0,0]

## generate probability of edges

for cn1,cn2 in g.edges:
    g[cn1][cn2]['weight'] = np.round(random.uniform(0.5, 1), 3)

## get weighted adjacency matrix of graph for map estimate

filepath = cnf.datapath + "\\cora_weighted.gpickle"
g = nx.read_gpickle(filepath)

for u,v in g.edges():
    g.edges[u,v]['weight'] = g.edges[u,v]['weight']

W = nx.linalg.graphmatrix.adjacency_matrix(g, dtype=np.float)
W = W.toarray()

mapping = dict(zip(g, range(0, len(g.nodes()))))
g_new = nx.relabel_nodes(g, mapping)

def getweightedadj_nxgraph(g):
    graph = g.copy()
    noofnodes = len(graph.nodes())
    wadj = np.zeros(shape=(noofnodes, noofnodes))
    edgelist = list(graph.edges())

    for iter in edgelist:
        try:
            wadj[iter[0],iter[1]] = graph.edges[iter[0], iter[1]]['weight']
            wadj[iter[1], iter[0]] = wadj[iter[0],iter[1]]
        except:
            print("nw")

    return wadj

w_adj_cora = getweightedadj_nxgraph(g_new)

tempdic = {}
tempdic['wadjacency'] = w_adj_cora
filename = cnf.datapath + "\\cora_weighted_adjacency.mat"
io.savemat(filename, tempdic)

filepath = cnf.datapath + "\\cora_weighted.pickle"

with open(filepath, 'rb') as f:
    g2 = pickle.load(f)

u,v = g2.all_edges(order='eid')

wadj=np.zeros((2708, 2708))

for count,cu,cv in enumerate(zip(u,v)):
    wadj[u,v] = g2.edata['weight'][th.tensor([count])].numpy()
    break


