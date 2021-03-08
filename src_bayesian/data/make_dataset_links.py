from src.data import utils as ut
from src.data import config
from src.features import build_features as bf
import pickle
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import numpy as np

## generate training data

graphsizelist = [200, 200, 500, 500, 600, 600, 700, 700, 800, 800]
graphsizelist = [1000, 1200]

fileext = "\link_pl_"+ str(sum(graphsizelist))

## generate egr score for links in synthetic graphs

md = ut.Genlinkdata()

def get_plc_linkegrscores(graphsizelist):
    Listgraph = []
    Listlabel = []

    for gsize in graphsizelist:

        g = nx.generators.random_graphs.powerlaw_cluster_graph(gsize, 2, 0.05)
        # g = nx.barabasi_albert_graph(n=gsize, m=3)

        linklist = list(g.edges())

        print("len edges", len(linklist))

        egrdict = md.get_linkegr(g, linklist)

        Listgraph.append(g)
        Listlabel.append(egrdict)

        print("gsize", gsize)

    return Listlabel, Listgraph

def get_pl_linkegrscores(graphsizelist):
    Listgraph = []
    Listlabel = []

    for gsize in graphsizelist:

        g1 = nx.scale_free_graph(gsize)
        g = nx.Graph(g1)
        g.remove_edges_from(nx.selfloop_edges(g))

        linklist = list(g.edges())
        print("len edges", len(linklist))

        egrdict = md.get_linkegr(g, linklist)

        Listgraph.append(g)
        Listlabel.append(egrdict)

        print("gsize", gsize)

    return Listlabel, Listgraph

def get_plcgraph_linkwsscores(graphsizelist):
    Listgraph = []
    Listlabel = []
    for gsize in graphsizelist :

        g = nx.generators.random_graphs.powerlaw_cluster_graph(gsize, 2, 0.05)
        linklist = list(g.edges())
        print("len edges", len(linklist))
        wsdict = md.get_linkws(g, linklist)
        Listgraph.append(g)
        Listlabel.append(wsdict)
        print(gsize)

    return Listlabel, Listgraph

def get_bagraph_linkwsscores(graphsizelist):
    Listgraph = []
    Listlabel = []
    for gsize in graphsizelist:

        g = nx.barabasi_albert_graph(n=gsize, m=3)

        linklist = list(g.edges())
        print("len edges", len(linklist))
        wsdict = md.get_linkws(g, linklist)
        Listgraph.append(g)
        Listlabel.append(wsdict)
        print(gsize)

    return Listlabel, Listgraph

Listlabel, Listgraph = get_plcgraph_linkwsscores(graphsizelist)

Listlabel, Listgraph = get_bagraph_linkwsscores(graphsizelist)

Listlabel, Listgraph = get_plc_linkegrscores(graphsizelist)

Listlabel, Listgraph = get_pl_linkegrscores(graphsizelist)

with open(config.datapath + 'Link'+ fileext + "_score.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

with open(config.datapath + 'Link'+ fileext + "_listgraph.pickle", 'wb') as b:
    pickle.dump(Listgraph, b)

## ================= load egr and ws score of graph

with open(config.datapath + 'Link' + fileext + "_egrscore.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

with open(config.datapath + 'Link' + fileext + "_listgraph.pickle", 'rb') as b:
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

link_ids = np.array(list(g.edges()))

def get_concatlist(Listranks):

    link_labels = Listranks[0]

    for countlen in range(len(Listranks) - 1):
        link_labels = np.concatenate((link_labels, Listranks[countlen + 1]), axis=0)

    return link_labels

link_labels = get_concatlist(Listranks)

### feature vector

for node_id, node_data in g.nodes(data=True):
    node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1,1]

##

