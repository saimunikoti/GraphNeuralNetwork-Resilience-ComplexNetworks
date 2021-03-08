import networkx as nx
import numpy as np
import pickle

fileext = "cit-DBLP.txt"

def get_graphtxt(path):
    g = nx.read_edgelist(path, create_using=nx.DiGraph(), nodetype=int)
    return g

g = get_graphtxt(fileext)
g = g.to_undirected()

gnodes = list(g.nodes())
nodelist = gnodes[8000:10000]

## get egr score
def get_egrdict(g, nodelist):

    def get_egr(graph):
        eig = nx.linalg.spectrum.laplacian_spectrum(graph, weight='weight')

        try:
            eigtemp1 = [1/num for num in eig if num > 5e-10]
            # eig = [1 / num for num in eig[1:] if num != 0]
            eigtemp2 = sum(np.abs(eigtemp1))
        except:
            print("zero encountered in Laplacian eigen values")

        Rg = (2 / (len(graph.nodes()) - 1))*(eigtemp2)
        return np.round(Rg, 3)

    egr_new = np.zeros(len(nodelist))

    for countnode, node in enumerate(nodelist):
        gcopy = g.copy()
        gcopy.remove_node(node)
        egr_new[countnode] = get_egr(gcopy)
        print(countnode)

    return egr_new

egrscore = get_egrdict(g, nodelist)

## get ws score

def get_weightedspectrum(graph):
    lambdas = nx.linalg.spectrum.normalized_laplacian_spectrum(graph)
    wghtspm = sum([(1 - eigs) ** 2 for eigs in lambdas])

    return round(wghtspm, 3)

def get_wghtspectnode(g):
    ws_new = np.zeros(len(g.nodes))

    for countnode, node in enumerate(g.nodes()):
        gcopy = g.copy()
        gcopy.remove_node(node)
        ws_new[countnode] = get_weightedspectrum(gcopy)

    return ws_new

def get_plcgraph_wsscores(graphsizelist):
    Listgraph = []
    Listlabel = []
    for gsize in graphsizelist :

        g = nx.generators.random_graphs.powerlaw_cluster_graph(gsize, 4, 0.05)
        egrdict = get_wghtspectnode(g)
        Listgraph.append(g)
        Listlabel.append(egrdict)
        print(gsize)

    return Listlabel, Listgraph

graphsizelist = [100, 200, 400, 400, 500, 500, 800, 800, 1000, 1000, 2000, 2000, 3000, 4000]
Listlabel, Listgraph = get_plcgraph_wsscores(graphsizelist)

fileext = "\ws_plc_"+ str(sum(graphsizelist))

with open(fileext + "_wsscore.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

with open(fileext + "_listgraph.pickle", 'wb') as b:
    pickle.dump(Listgraph, b)
