import networkx as nx
from src.data import utils as ut
from src.data import config
from src.features import build_features as bf
import pickle
import scipy.io as io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
## ================== generate observed graph and label for multiple graphs =========================

# Listgraph, Listlabel = ut.get_weightedgraphfeaturelabel_syn('plc', 'egr', [2000])
# gobs = bf.combine_graphs(Listgraph)

# ## ================= save graph and corresponding labels features ===============
# #
# fileext = "\\plc_2000_egr_bayesian"
#
# nx.write_gpickle(gobs, config.datapath + 'Bayesian'+fileext + ".gpickle")
#
# with open(config.datapath + 'Bayesian'+ fileext + "_label.pickle", 'wb') as b:
#     pickle.dump(Listlabel, b)

## ================ generate single graph feature and label weighted =====================

fileext = "\\plc_3000_egr"

gobs = nx.generators.random_graphs.powerlaw_cluster_graph(3000, 1, 0.1)
nx.write_gpickle(gobs, config.datapath +"processed" + fileext + ".gpickle")

Listlabel = []
Listlabel.append(ut.get_estgraphlabel(gobs, "egr", weightflag=1))


with open(config.datapath + 'Bayesian'+ fileext + "_label.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

## ================ load graph and load label ==============

gobs = nx.read_gpickle(config.datapath + 'Bayesian'+ fileext+".gpickle")

with open(config.datapath + 'Bayesian' + fileext+ "_label.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

## === generate target vector by combining labels from all graphs and keep it in target data frame ============
# specific nodes and columns
gobs.remove_node(0)
nodelist = list(gobs.nodes)

targetdf = bf.getgraphtargetdf(Listlabel, nodelist)
plt.hist(targetdf['metric'])
category = pd.cut(targetdf.metric, bins=[0,0.3,0.7,1.0],labels=[0, 1, 2])
targetdf['metric'] = category
plt.hist(targetdf['metric'])
targetdf = targetdf.dropna()
targetdf = pd.get_dummies(targetdf.metric)

# noderanks = np.argsort(np.array(targetdf['metric']))
# top10nodes = targetdf.loc[noderanks, 'metric']

## ====================== assign feature vector to each graph node ==========================

bf.get_graphnodefeatures(gobs)

## random noisy graph v1 for evaluating model performance
# gobscopy = gobs.copy()
# edgelist = list(gobs.edges())
# randomedgelist = random.sample(edgelist, int(0.05*len(edgelist)) )
# gobscopy.remove_edges_from(randomedgelist)
# # count = 0
# countoutloop = 0
# for a in gobscopy.nodes():
#     for b in gobscopy.nodes():
#         if a != b and (a, b) not in edgelist:
#             gobscopy.add_edge(a, b)
#             count += 1
#
#             if count > int(0.05*len(edgelist)):
#                 countoutloop =1
#                 break
#
#     if countoutloop == 1:
#         break

# fileext = "\\plc_5000_gobsnoise_bayesian"
# nx.write_gpickle(gobscopy, config.datapath + 'Bayesian'+fileext + ".gpickle")

## noisy graph v2
gobscopy = gobs.copy()
nodelist = list(gobs.nodes())
randomnodelist = random.sample(nodelist, int(0.05*len(nodelist)) )

def knbrs(G, start, k):
    nbrs = set([start])
    nothop1nbrs = set()
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
        if l>0:
            for val in nbrs:
                nothop1nbrs.add(val)

    return nbrs, nothop1nbrs
# add edges

for a in randomnodelist :
    nbrs, nothop1nbrs = knbrs(gobscopy, a, 2)
    nbrs = nbrs-set([a])
    newedge = (a, random.sample(nbrs, 1)[0])
    gobscopy.add_edge(newedge[0], newedge[1] )

print(len(gobscopy.edges))
fileext = "\\plc_5000_gobsnoiseadd_bayesian"
nx.write_gpickle(gobscopy, config.datapath + 'Bayesian'+fileext + ".gpickle")

# remove edges

for a in randomnodelist:
    tempedges = list(gobs.edges(a))
    edgeremoved = random.sample(tempedges, 1)[0]
    # count=0
    # while count==0:
    #     if edgelist in newedgelist:
    #         edgelist = random.sample(tempedges, 1)[0]
    #         print("----")
    #     else:
    #         gobscopy.remove_edges_from([edgelist])
    #         count=1
    #         break
    gobscopy.remove_edges_from([edgeremoved])

print(len(gobscopy.edges))

fileext = "\\plc_5000_gobsnoiseaddremoved_bayesian"
nx.write_gpickle(gobscopy, config.datapath + 'Bayesian'+fileext + ".gpickle")

fileext = "\\plc_5000_gobsnoise_bayesian"
with open(config.datapath + 'Bayesian'+ fileext + "_nodelist.pickle", 'wb') as b:
    pickle.dump(randomnodelist, b)

## load noisy version
fileext = "\\plc_5000_gobsnoiseaddremoved_bayesian"

gobsnoise = nx.read_gpickle(config.datapath + 'Bayesian'+ fileext+".gpickle")

with open(config.datapath + 'Bayesian'+ fileext + "_nodelist.pickle", 'rb') as b:
    randomnodelist = pickle.load(b)

## get adjacency Matrix for edge mask
fileext = "\\plc_5000_egr_bayesian"

with open(config.datapath + 'Bayesian'+ fileext + "_nodelist.pickle", 'rb') as b:
    nodelist = pickle.load(b)

gobs = nx.read_gpickle(config.datapath + 'Bayesian' + fileext+".gpickle")

# get new adjacenency matrix for edge mas
# gmask = gobs.copy()
# nx.is_connected(gmask)
#
# for a in randomnodelist:
#     nbrs, nothop1nbrs = knbrs(gobs, a, 2)
#     nbrs = list(nbrs-set([a]))
#     newedgelist = [(a, nbrs[ind]) for ind in range(len(nbrs))]
#     gmask.add_edges_from(newedgelist)

## get new adjacency matrix fopr edge masks
gmask = gobs.copy()
nx.is_connected(gmask)

for a in randomnodelist:

    allnbrs = nx.single_source_shortest_path_length(gobs, a, cutoff=4)
    nothop1nbrs = [key for (key,value) in allnbrs.items() if allnbrs[key]>1 ]
    selnbrs = np.random.choice(nothop1nbrs,4,replace=False)
    newedgelist = [(a, selnbrs[ind]) for ind in range(len(selnbrs))]
    gmask.add_edges_from(newedgelist)

## ============= save weighted adjacency matrix mask for esimnating new adjacency matrix from GSP ===================

W = nx.adjacency_matrix(gmask, weight='weight')
W = W.toarray()
A = nx.adjacency_matrix(gmask, weight=None)
A = A.toarray()

tempdic = {}
tempdic['wadjacency'] = W
tempdic['adjacency'] = A
filename = config.datapath + "Bayesian"+ fileext+ "_wadjacency.mat"
io.savemat(filename, tempdic)

## noise version 3
gobsnoise = gest.copy()
nx.is_connected(gobsnoise)

for a in randomnodelist[1:100]:

    allnbrs = nx.single_source_shortest_path_length(gest, a, cutoff=4)
    nothop1nbrs = [key for (key,value) in allnbrs.items() if allnbrs[key]>1 ]
    selnbrs = np.random.choice(nothop1nbrs,2,replace=False)
    newedgelist = [(a, selnbrs[ind]) for ind in range(len(selnbrs))]
    gobsnoise.add_edges_from(newedgelist)

print("", len(gobsnoise.edges))
print("", len(gobsnoise.nodes))

fileext = "\\plc_5000_gobsnoiseadd_bayesian_6195"
nx.write_gpickle(gobsnoise, config.datapath + 'Bayesian'+fileext + ".gpickle")

## noisy version 4 removing edges
gobsnoise = gobs.copy()
nx.is_connected(gobsnoise)

for a in randomnodelist:
    allnbrs= list(gobsnoise.edges(a))
    selnbrs = random.sample(allnbrs, 1)
    gobsnoise.remove_edges_from(selnbrs)

print(len(gobsnoise.edges))
print(len(gobsnoise.nodes))

fileext = "\\plc_5000_gobsnoise_edgeremoved_bayesian"
nx.write_gpickle(gobsnoise, config.datapath + 'Bayesian'+fileext + ".gpickle")