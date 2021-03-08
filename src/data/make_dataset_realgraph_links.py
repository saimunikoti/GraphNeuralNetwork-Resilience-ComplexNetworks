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

## ==================== aggregating multiple files for wiki vote

fileext = "wiki-vote_100-107k_link_ws_score.pickle"
filepath = config.datapath + "link\\" + fileext

with open(filepath, 'rb') as b:
    egrscore5 = pickle.load(b)

egrscorenet = list(itertools.chain(egrscore[0], egrscore2[0], egrscore3[0], egrscore4[0], egrscore5[0]))

##
fileext ="wiki-vote_0-25k_link_ws_listgraph.pickle"
filepath = config.datapath + "link\\" + fileext

with open(filepath, 'rb') as b:
    g1 = pickle.load(b)

## ======= generate graph and corresponding labels for real world networks
g = ut.get_graphtxt(filepath)
g = g.to_undirected()
gnodes = list(g.nodes())
nodelist = gnodes

# get egr score
md = ut.GenEgrData()
egrscore = md.get_egrdict(g, nodelist)

##======================= save egr score of real graph

with open(config.datapath + 'Link\\'+ fileext + "_egrscore.pickle", 'wb') as b:
    pickle.dump(egrscore, b)

with open(config.datapath + 'Link\\'+ fileext + ".gpickle", 'wb') as b:
    pickle.dump(g, b)

## ================= load egr score of graph

fileext = "bio-yeastlink_egr"

with open(config.datapath + 'Link\\' + fileext + "_score.pickle", 'rb') as b:
    egrscore = pickle.load(b)

with open(config.datapath + 'Link\\' + fileext + ".pickle", 'rb') as b:
    g = pickle.load(b)

g = g[0]

egrdict = rankdata(egrscore, method='dense')

link_ids = np.array(list(g.edges()))
link_labels = (egrdict - min(egrdict)) / (max(egrdict) - min(egrdict))

plt.figure(1)
plt.plot(link_labels)
plt.hist(link_labels)
plt.close(1)

### feature vector

for node_id, node_data in g.nodes(data=True):
    node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1,1]
    # node_data["feature"] = [g.degree(node_id), 1, 1,1]




##

