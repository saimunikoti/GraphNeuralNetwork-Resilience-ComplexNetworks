
import networkx as nx
from src_bgnn.data import utils as ut
from src_bgnn.data import config as cnf
from src_bgnn.features import build_features as bf
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

## generate single graph and label for egr score

fileext = "\\plc_1000"
g = nx.generators.random_graphs.powerlaw_cluster_graph(4000, 1, 0.1)
Listlabel = []
Listlabel.append(ut.get_estgraphlabel(g, "egr", weightflag=1))

## generate Multiple graphs and labels for egr

graphsizelist = [100, 200, 400, 400, 500, 500, 800, 800, 1000, 1000, 2000, 2000, 3000, 4000]

fileext = "\plc_"+ str(sum(graphsizelist))

Listgraph = []
for countsize in graphsizelist:
    Listgraph.append(nx.generators.random_graphs.powerlaw_cluster_graph(countsize, 1, 0.1))

Listlabel = []
for countg in Listgraph:
    print("--")
    Listlabel.append(ut.get_estgraphlabel(countg, "egr", weightflag=1))

# combine graphs into one disjoint union graph
g = bf.combine_graphs(Listgraph)

with open(cnf.datapath + fileext + "_Listgraph.pickle", 'wb') as b:
    pickle.dump(Listgraph, b)

with open(cnf.datapath + fileext + "_Listlabel.pickle", 'wb') as b:
     pickle.dump(Listlabel, b)

## ================ load graph and node-labels ==============

fileext = "\\plc_16700"
g = nx.read_gpickle(cnf.datapath + fileext + ".gpickle")

with open(cnf.datapath + fileext + "_Listgraph.pickle", 'rb') as b:
    Listgraph = pickle.load(b)

with open(cnf.datapath + fileext + "_Listlabel.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

g = Listgraph[0]

Labelarray = Listlabel[0]

## === generate target vector by combining labels from all graphs and keep it in target data frame ============

# generate target vector for list of graphs
nodelist = list(np.arange(0, len(g.nodes)))

targetdf = bf.getgraphtargetdf(Listlabel, nodelist)

category = pd.cut(targetdf.metric,  bins=[0.0,0.3,0.7,1.0], labels=[0,1,2], include_lowest=True)
targetdf['metric'] = category
plt.hist(targetdf['metric'])

# targetdf = pd.get_dummies(targetdf.metric)

# ==== generate target vector for single graph
# nodelist = list(g.nodes)
# # gen datagrame of target labels
# targetdf = pd.DataFrame()
# targetdf['metric'] = Labelarray
# targetdf['nodename'] = nodelist
# targetdf = targetdf.set_index('nodename')
#
# # targetdf = bf.getgraphtargetdf(Listlabel, nodelist)
# # # targetdf.loc[targetdf.metric==0,'metric'] = 0.001
#
# category = pd.cut(targetdf.metric,  bins=[0.0,0.7,1.0], labels=[0, 1], include_lowest=True)
# targetdf['metric'] = category
# plt.hist(targetdf['metric'])

## assign node label to gobs

for node_id, node_data in g.nodes(data=True):
    node_data["label"] = list(targetdf.loc[node_id])

## ====================== assign feature vector to each graph node ==========================

bf.get_graphnodefeatures(g)

# save gpobs with feature and node labes

filepath = cnf.datapath + fileext + ".gpickle"

nx.write_gpickle(g, filepath)

## Generate random instances of graph

Listgraph = bf.generate_multiple_graphinstance(g, 10000)

fileext = "\plc_10700_instances"

with open(cnf.datapath + fileext + ".pickle", 'wb') as b:
    pickle.dump(Listgraph, b)

## GENERATE VARIANCE of each node feature

fileext = "\plc_10700_instances"

with open(cnf.datapath + fileext + ".pickle", 'rb') as b:
    Listgraph = pickle.load(b)

Meandic, Vardic = bf.get_feature_meanandvariance(g, Listgraph)

for node_id, node_data in g.nodes(data=True):
    node_data["meanfeature"] = Meandic[node_id].tolist()
    node_data["varfeature"] = Vardic[node_id].tolist()

# save g with mean and variance feature and node labels

fileext = "\\plc_10700"
filepath = cnf.datapath + fileext + ".gpickle"

nx.write_gpickle(g, filepath)





