from src.data import utils as ut
from src.data import config
from src.features import build_features as bf
import pickle
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from scipy.stats import rankdata
import numpy as np
## ===================== generate egr score for us power grid graph ===============
fileext = "\Wiki-Vote.txt"
fileext = "\power-US-Grid.txt"
fileext = "ca-netscience.txt"
fileext = "bio-yeast.txt"
fileext = "cit-DBLP.txt"

filepath = config.datapath + "raw\\" + fileext
# generate graph and corresponding labels for real world networks
g = ut.get_graphtxt(filepath)
g = g.to_undirected()
gnodes = list(g.nodes())
nodelist = gnodes

# get egr score
md = ut.GenEgrData()
egrscore = md.get_egrdict(g, nodelist)

# get ws score
def get_realgraph_wsscores(filepath):
    Listgraph = []
    Listlabel = []
    g = ut.get_graphtxt(filepath)
    g = g.to_undirected()

    wsdict = md.get_wghtspectnode(g)
    Listgraph.append(g)
    Listlabel.append(wsdict)

    return Listlabel, Listgraph

Listlabel, Listgraph = get_realgraph_wsscores(filepath)

##======================= save egr score of real graph

with open(config.datapath + 'processed\\' + "US-powergrid" + "node_ws"+"_score.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

with open(config.datapath + 'processed\\' + "US-powergrid" + "node_ws" + "_listgraph.pickle", 'wb') as b:
    pickle.dump(Listgraph, b)

## ================= load egr score of graph

with open(config.datapath + 'processed\\' + fileext+ "_egrscore.pickle", 'rb') as b:
    egrscore = pickle.load(b)

with open(config.datapath + 'processed\\' + fileext+ ".gpickle", 'rb') as b:
    g = pickle.load(b)

egrdict = rankdata(egrscore, method='dense')
egrranknorm = (egrdict - min(egrdict)) / (max(egrdict) - min(egrdict))

## generate target vector for syn graph
Listlabelnew =[]
Listlabelnew.append(egrranknorm)


targetdf = bf.getgraphtargetdf(Listlabelnew, nodelist)
plt.figure(1)
plt.plot(targetdf['metric'])
plt.hist(targetdf.metric)
plt.close(1)
### feature vector

for node_id, node_data in g.nodes(data=True):
    node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1,1]
    # node_data["feature"] = [g.degree(node_id), 1, 1,1]



