import networkx as nx
import numpy as np
import pickle
from src.data import config

# fileext = "Wiki-Vote.txt"
# fileext = "power-US-Grid.txt"
# fileext = "ca-netscience.txt"
fileext = "bio-yeast.txt"
# fileext = "cit-DBLP.txt"

filepath = config.datapath + "raw\\" + fileext

def get_graphtxt(path):
    g = nx.read_edgelist(path, create_using=nx.DiGraph(), nodetype=int)
    return g

g = get_graphtxt(fileext)
g = g.to_undirected()

gnodes = list(g.nodes())
nodelist = gnodes[8000:10000]

# get egr score
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


with open("egrdblp_8to10k.pickle", 'wb') as b:
    pickle.dump(egrscore, b)
    
    
#%%
## get ws score for nodes

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

def get_linkws(g, linklist):

    ws_new = np.zeros(len(linklist))

    for countlink, link in enumerate(linklist):
        gcopy = g.copy()
        gcopy.remove_edge(*link)
        ws_new[countlink] = get_weightedspectrum(gcopy)
        print(countlink)

    return ws_new

#%%
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

#%% ws scores for links

def get_plcgraph_linkwsscores(graphsizelist):
    Listgraph = []
    Listlabel = []
    
    for gsize in graphsizelist :

        g = nx.generators.random_graphs.powerlaw_cluster_graph(gsize, 2, 0.05)
        linklist = list(g.edges())
        print("len edges", len(linklist))
        wsdict = get_linkws(g, linklist)
        Listgraph.append(g)
        Listlabel.append(wsdict)
        print(gsize)

    return Listlabel, Listgraph

def get_realgraph_linkwsscores(filepath):
    Listgraph = []
    Listlabel = []

    g = get_graphtxt(filepath)
    g = g.to_undirected()
    
    linklist = list(g.edges())
    linklist = linklist[50000:80000]
    
    print("len edges", len(linklist))
    wsdict = get_linkws(g, linklist)
    Listgraph.append(g)
    Listlabel.append(wsdict)


    return Listlabel, Listgraph

Listlabel, Listgraph = get_plcgraph_linkwsscores(graphsizelist)

Listlabel, Listgraph = get_realgraph_linkwsscores(filepath)
  
with open("USpowergrid_" + "link_ws"+"_score.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

with open("USpowergrid_" + "link_ws"+ "_listgraph.pickle", 'wb') as b:
    pickle.dump(Listgraph, b)
    
 #%% egr score link scores
 
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

def get_linkegr(g, linklist):
    
    egr_new = np.zeros(len(linklist))

    for countlink, link in enumerate(linklist):
        gcopy = g.copy()
        gcopy.remove_edge(*link)
        egr_new[countlink] = get_egr(gcopy)
        print(countlink)

    return egr_new

def get_real_linkegrscores(filepath):
    Listgraph = []
    Listlabel = []

    g = get_graphtxt(filepath)
    g = g.to_undirected()

    linklist = list(g.edges())

    print("len edges", len(linklist))

    egrdict = get_linkegr(g, linklist)

    Listgraph.append(g)
    Listlabel.append(egrdict)



    return Listlabel, Listgraph

def get_plc_linkegrscores(graphsizelist):
    Listgraph = []
    Listlabel = []

    for gsize in graphsizelist:

        g = nx.generators.random_graphs.powerlaw_cluster_graph(gsize, 3, 0.05)
        # g = nx.barabasi_albert_graph(n=gsize, m=3)

        linklist = list(g.edges())

        print("len edges", len(linklist))

        egrdict = get_linkegr(g, linklist)

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

        egrdict = get_linkegr(g, linklist)

        Listgraph.append(g)
        Listlabel.append(egrdict)

        print("gsize", gsize)

    return Listlabel, Listgraph   

def get_wast_linkegrscores(graphsizelist):
    Listgraph = []
    Listlabel = []

    for gsize in graphsizelist:

        g = nx.connected_watts_strogatz_graph(n=gsize, k=6, p=0.1)
        
        linklist = list(g.edges())
        print("len edges", len(linklist))

        egrdict = get_linkegr(g, linklist)

        Listgraph.append(g)
        Listlabel.append(egrdict)

        print("gsize", gsize)

    return Listlabel, Listgraph 

#%%    
graphsizelist = [200, 200, 500, 500, 600, 600, 700, 700, 800, 800]
    
Listlabel, Listgraph = get_wast_linkegrscores(graphsizelist)

fileext = "power-US-Grid_egr"

Listlabel, Listgraph = get_real_linkegrscores(filepath)

with open(config.datapath + 'Link\\'+ fileext + "_score.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

with open(config.datapath + 'Link\\'+ fileext + "_listgraph.pickle", 'wb') as b:
    pickle.dump(Listgraph, b)
