
import numpy as np
import networkx as nx
g = nx.generators.random_graphs.powerlaw_cluster_graph(50, 1, 0.05)

#### effective graph resistance from eigen values of laplacian
def get_egr(graph):
    eig = nx.linalg.spectrum.laplacian_spectrum(graph)
    try:
        # eig = [1/num for num in eig if num>0.00005]
        eig = [1/num for num in eig[1:] if num != 0]
        eig = sum(np.abs(eig))
    except:
        print("zero encountered in Laplacian eigen values")
    Rg = (2/(len(graph.nodes())-1))*eig
    return np.round(Rg,3)

### get rank of edges from eigen values of laplacian method- exact method

def get_egrlinkrank(g):
    # egr_old = get_egr(g)

    egr_new = np.zeros(len(g.edges))
    for countedge, (v1,v2) in enumerate(g.edges()):
        g[v1][v2]['edgepos'] = countedge
        gcopy = g.copy()
        gcopy.remove_edge(v1,v2)
        egr_new[countedge] = get_egr(gcopy)

    # egr_diff = egr_new - egr_old
    order = egr_new.argsort()
    ranks = order.argsort() + 1  # getting ranks from 1 and high rank denotes high importance score

    return egr_new, ranks

egr_new , edgeranks1 = get_egrlinkrank(g)
print(edgeranks1)
print(egr_new)

### get rank of edges from effective resistance mehtod -approximation - low time complexity.
def get_approx_egrlinkrank(g):
    L = nx.laplacian_matrix(g).todense()
    LInv = np.linalg.pinv(L)
    Rl = np.zeros(len(g.edges))
    # Rd = np.zeros(len(g.edges))
    for countedge, (v1, v2) in enumerate(g.edges()):
        g[v1][v2]['edgepos'] = countedge
        Rl[countedge] = LInv[v1,v1] + LInv[v2,v2] -2*LInv[v1,v2]
        # Rd[countedge] = nx.algorithms.distance_measures.resistance_distance(g,v1,v2)
    order = Rl.argsort()
    ranks_rl = order.argsort() + 1
    # order = Rd.argsort()
    # ranks_rd = order.argsort() + 1

    return ranks_rl #, ranks_rd

edgeranks2 = get_approx_egrlinkrank(g)
print(edgeranks2)

nx.draw(g, pos=nx.spring_layout(g), with_labels=True, node_color="bisque")
edge_labels = nx.get_edge_attributes(g,'edgepos')
nx.draw_networkx_edge_labels(g, pos=nx.spring_layout(g), labels = edge_labels)
edge_labels=dict([((u,v,),d['edgepos']) for u,v,d in g.edges(data=True)])



