import networkx as nx
from src.data import utils as ut
from src.data import config
from src.features import build_features as bf
import pickle
import scipy.io as io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
## ================== generate observed graph and label =========================

# Listgraph, Listlabel = ut.get_weightedgraphfeaturelabel_syn('plc', 'egr', [2000])
#
# gobs = bf.combine_graphs(Listgraph)
#
# ## ================= save graph and corresponding labels features ===============
# #
# fileext = "\\plc_2000_egr_bayesian"
#
# nx.write_gpickle(gobs, config.datapath + 'Bayesian'+fileext + ".gpickle")
#
# with open(config.datapath + 'Bayesian'+ fileext + "_label.pickle", 'wb') as b:
#     pickle.dump(Listlabel, b)

fileext = "\\plc_2000_egr_bayesian"

gobs = nx.generators.random_graphs.powerlaw_cluster_graph(2000, 1, 0.1)

Listlabel = []
Listlabel.append(ut.get_estgraphlabel(gobs, "egr"))

with open(config.datapath + 'Bayesian'+ fileext + "_label.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

## ================ load graph and load label ==============

gobs = nx.read_gpickle(config.datapath + 'Bayesian'+ fileext+".gpickle")

with open(config.datapath + 'Bayesian' + fileext+ "_label.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

## === generate target vector by combniing labels from all graphs and keep it in target data frame ============

targetdf = bf.getgraphtargetdf(Listlabel, gobs)
plt.hist(targetdf['metric'])
category = pd.cut(targetdf.metric, bins=[0,0.25,0.7,1.0],labels=[0, 1, 2])
targetdf['metric'] = category
plt.hist(targetdf['metric'])
targetdf = pd.get_dummies(targetdf.metric)

# noderanks = np.argsort(np.array(targetdf['metric']))
# top10nodes = targetdf.loc[noderanks, 'metric']

## ====================== assign feature vector to each graph node ==========================

bf.get_graphnodefeatures(gobs)

## ============= save weighted adjacency matrix ===================

W = nx.adjacency_matrix(gobs, weight='weight')
W = W.toarray()

A = nx.adjacency_matrix(gobs, weight=None)
A = A.toarray()

tempdic = {}
tempdic['wadjacency'] = W
tempdic['adjacency'] = A
filename = config.datapath + "Bayesian"+ fileext+ "_wadjacency.mat"
io.savemat(filename, tempdic)

