import pickle
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph.core.graph import StellarGraph

from src_bgnn.data import config as cnf
from src_bgnn.data import utils as ut
from src_bgnn.features import build_features as bf

## generate random plc ba graph

# graphsizelist = [200, 300, 500, 500, 600, 600, 700, 700, 800, 800]
graphsizelist = [4000]

sumgraphsizelist = list(np.cumsum(graphsizelist))
sumgraphsizelist.insert(0, 0)

fileext = "\plc_"+ str(sum(graphsizelist))

filepath = cnf.datapath + fileext

Listgraph = []
for countsize in graphsizelist:
    Listgraph.append(nx.generators.random_graphs.powerlaw_cluster_graph(countsize, 1, 0.1))

Listlabel = []
for countg in Listgraph:
    print("--")
    Listlabel.append(ut.get_estgraphlabel(countg, "egr", weightflag=1))

with open(cnf.datapath + fileext + "_label.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

with open(cnf.datapath + fileext + "_listgraph.pickle", 'wb') as b:
    pickle.dump(Listgraph, b)

## load test protein graph
filepath = cn.datapath + "\\ppi\ppi-G.csv"

g = ut.get_graphfromdf(filepath, 'source', 'target')

filepath = cn.datapath + "\\ppi\\ppi-class_map.json"

targets = ut.get_jsondata(filepath)

mapper = {}
for count, nodeid in enumerate(g.nodes):
    mapper[nodeid] = count

g = nx.relabel_nodes(g, mapper)
newdict={}
for count,(key,value) in enumerate(targets.items()):
    newdict[mapper[int(key)]] = value
    # print(key, value)
    # if count>3:
    #     break

targets = pd.DataFrame.from_dict(newdict, orient='index')

#========================= load egr graph===========================

with open(cnf.datapath + fileext + "_label.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

with open(cnf.datapath + fileext + "_listgraph.pickle", 'rb') as b:
    Listgraph = pickle.load(b)

# remove node 0 from graph and Labels
Listgraphnew = []
Listlabelnew = []

for count in range(len(Listgraph)):
    graphcopy = Listgraph[count].copy()
    graphcopy.remove_node(0)
    mapping = {k:k-1 for k in range(1, len(graphcopy.nodes)+1 )}
    H = nx.relabel_nodes(graphcopy, mapping)
    Listgraphnew.append(H)
    Listlabelnew.append(Listlabel[count][1:])

# combine graphs into one disjoint union graph
g = bf.combine_graphs(Listgraphnew)

# generate target vector for syn graph

nodelist = list(np.arange(0, len(g.nodes)))

targetdf = bf.getgraphtargetdf(Listlabelnew, nodelist)
plt.figure(1)
plt.plot(targetdf['metric'])
plt.hist(targetdf.metric)
plt.close(1)

category = pd.cut(targetdf.metric,  bins=[0,0.3,0.7,1.0], labels=[0, 1, 2])
targetdf['metric'] = category
plt.hist(targetdf['metric'])
targetdf = pd.get_dummies(targetdf.metric)

## ====================== assign feature vector to each graph node ==========================

bf.get_graphnodefeatures(g)

## split data and get stellar graph

train_subjects, test_subjects = model_selection.train_test_split(targetdf, test_size=0.15)

train_targets = np.array(train_subjects)
test_targets = np.array(test_subjects)

G = StellarGraph.from_networkx(g, node_features="feature")
print(G.info())




##

