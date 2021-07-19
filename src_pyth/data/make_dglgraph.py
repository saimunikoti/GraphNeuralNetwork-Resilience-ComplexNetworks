
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

fileext = "\\plc_600_egr"

gobs = nx.generators.random_graphs.powerlaw_cluster_graph(600, 1, 0.1)

Listlabel = []
Listlabel.append(ut.get_estgraphlabel(gobs, "egr", weightflag=1))

filepath = config.datapath + 'dgl'+ fileext + "_label.pickle"

with open(filepath, 'wb') as b:
    pickle.dump(Listlabel, b)

## ================ load graph and load label ==============

fileext = "\\plc_5000_egr_bayesian"

gobs = nx.read_gpickle(config.datapath + 'dgl'+ fileext+".gpickle")

with open(config.datapath + 'dgl' + fileext+ "_label.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

Labelarray = Listlabel[0]
Labelarray = np.delete(Labelarray, [0])

## === generate target vector by combining labels from all graphs and keep it in target data frame ============
# specific nodes and columns

gobs.remove_node(2)
nodelist = list(gobs.nodes)
nodelist.pop(0)

# gen datagrame of target labels
targetdf = pd.DataFrame()
targetdf['metric'] = Labelarray
targetdf['nodename'] = nodelist
targetdf = targetdf.set_index('nodename')

# targetdf = bf.getgraphtargetdf(Listlabel, nodelist)
# # targetdf.loc[targetdf.metric==0,'metric'] = 0.001

category = pd.cut(targetdf.metric,  bins=[0,0.3,0.7,1.0], labels=[0, 1, 2])
targetdf['metric'] = category
plt.hist(targetdf['metric'])

## assign node label to gobs

for node_id, node_data in gobs.nodes(data=True):
    node_data["label"] = list(targetdf.loc[node_id])

## ====================== assign feature vector to each graph node ==========================

bf.get_graphnodefeatures(gobs)

# save gpobs with feature and node labes

filepath = config.datapath +"dgl" + fileext + ".gpickle"

nx.write_gpickle(gobs, filepath)

##






