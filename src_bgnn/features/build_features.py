import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def get_degreedist():
    btwlist=[]
    for countgraph in range(20000):
        g = nx.generators.random_graphs.powerlaw_cluster_graph(12, 1, 0.4)
        nodeimpscore = list(nx.betweenness_centrality(g).values())
        btwlist.extend(nodeimpscore)

    plt.hist(btwlist, bins=15, density=False, histtype='stepfilled', facecolor='g', alpha=0.3, label='Histogram')

def get_erbtwdist():
    btwlist=[]
    for countgraph in range(20000):
        g = nx.erdos_renyi_graph(22,0.2)
        nodeimpscore = list(nx.betweenness_centrality(g).values())
        btwlist.extend(nodeimpscore)

    plt.hist(btwlist, bins=15, density=False, histtype='stepfilled', facecolor='g', alpha=0.3, label='Histogram')

def get_albabtwdist():
    btwlist=[]
    deglist=[]
    for countgraph in range(20000):
        g = nx.barabasi_albert_graph(22,5)
        nodeimpscore = list(nx.betweenness_centrality(g).values())
        degimpscore = dict(g.degree()).values()
        deglist.extend(degimpscore)
        btwlist.extend(nodeimpscore)

    plt.hist(btwlist, bins=15, density=False, histtype='stepfilled', facecolor='g', alpha=0.3, label='Histogram')
    plt.figure(2)
    plt.hist(deglist, bins=15, density=False, histtype='stepfilled', facecolor='coral', alpha=0.3, label='Histogram')

def get_rgbtwdist():
    btwlist=[]
    deglist=[]
    for countgraph in range(20000):
        g = nx.random_geometric_graph(22, 0.2)
        nodeimpscore = list(nx.betweenness_centrality(g).values())
        degimpscore = dict(g.degree()).values()
        deglist.extend(degimpscore)
        btwlist.extend(nodeimpscore)

    plt.hist(btwlist, bins=15, density=False, histtype='stepfilled', facecolor='g', alpha=0.3, label='Histogram')
    plt.figure(2)
    plt.hist(deglist, bins=15, density=False, histtype='stepfilled', facecolor='coral', alpha=0.3, label='Histogram')

def get_ermodel():
    g = nx.erdos_renyi_graph(22, 0.2, seed=None, directed=False)
    print(nx.betweenness_centrality(g))
    pos = nx.circular_layout(g)
    nx.draw_networkx(g, pos)

def get_tranformpred(ypred):
    ypred[ypred>=1.5]=2
    ypred[(ypred < 1.5) & (ypred >0.5)] = 1
    ypred[ypred <= 0.5] = 0
    ypred = ypred.astype(int)
    return ypred

def inputvar(xtrain, xtest):
    count=0
    for data in xtrain:
        for testdata in xtest:
            if np.array_equal(testdata, data):
                count=count+1
    return count

## largest connected comp
def get_LCC(G, n):

    def simulate_lcc(graph):
        g= graph.copy()
        templcc = [len(xind) for xind in nx.connected_components(g)]

        while (len(templcc) ==1):

            nodeselected = np.random.choice(g.nodes())

            ##### node removal at primary level
            g.remove_node(nodeselected)

            #### effect of input on output- transitive propert

            #### collecting metrics
            templcc = [len(xind) for xind in nx.connected_components(g)]

        else:
            Lcc = max(templcc)

        return Lcc
    Lcc=[]
    for countsim in range(n):
        Lcc.append(simulate_lcc(G))

    return np.mean(Lcc)

#### variation of graph reistance with size and instances
def check_variation():
    for countn in [100]:
        t=[]
        for countg in range(2000):
            # G = nx.generators.random_graphs.powerlaw_cluster_graph(countn ,1, 0.05)
            # G = nx.barabasi_albert_graph(200, 3)
            G = nx.erdos_renyi_graph(countn,0.3)
            t.append(nx.current_flow_closeness_centrality(G))
            # t.append(get_egr(G))
        # meanegr.append(np.mean(t))
        # egrvar.append(np.var(t))

    plt.plot(t)
    plt.hist(t, bins=15)

### checking eff resistance between vertices and resistance distance is equal or not
def check_effres(g):

    L = nx.laplacian_matrix(g).todense()
    LInv = np.linalg.pinv(L)
    ea = np.zeros((len(g.nodes),1))
    ea[10]=1
    eb = np.zeros((len(g.nodes),1))
    eb[15]=1
    Rab = np.matmul(np.matmul((ea-eb).T,LInv),(ea-eb))

    Rabdist = nx.algorithms.distance_measures.resistance_distance(g,10,15)

    return Rab, Rabdist

### egg graph resistance from resistance distance ---high time complexity
def get_effgraphres(g):
    egr = 0
    N = len(g.nodes)
    for i in range(0,N):
        for j in range(i+1,N):
            egr = egr + nx.algorithms.distance_measures.resistance_distance(g,i,j)

    return egr

#### effective graph resistance from eigen values of laplacian
def get_egr(graph):
    eig = nx.linalg.spectrum.laplacian_spectrum(graph)
    eig = [1/num for num in eig[1:]]
    eig = np.sum(np.array(eig))
    # Rg = len(graph.nodes())*eig
    # Rg_n = (2/(len(graph.nodes())-1))*eig
    return (len(graph.nodes)*eig)

def get_nodefeature(g):

    X = np.ones(shape=(len(g.nodes), 3))
    X[:, 0] = np.sum(nx.adjacency_matrix(g).todense(), axis=1)

    return X

### expand node pair combination
def expandy(v, factor):
    temp = np.arange(0, v)
    indexarray = np.zeros((factor*v, 2), dtype=int)
    indexarray[:, 0] = np.random.choice(temp, factor * v)
    indexarray[:, 1] = np.random.choice(temp, factor * v)
    indexselected = []
    for ind1 in range(factor * v):
        if indexarray[ind1, 0] != indexarray[ind1, 1]:
            indexselected.append(ind1)

    # return np.unique(indexarray[indexselected, :], axis=0)
    return indexarray[indexselected, :]

def classifylabels(a):
    a = np.where(a <= 0.33, 0, a)
    a = np.where(a >= 0.66, 2, a)
    a = np.where((a > 0.33) & (a < 0.66), 1, a)

    return a

###### loss funtions for node egr loss
def noderankloss(index):

    def loss(y_true, y_pred):
        # tf.print(tf.gather(y_true, tf.constant(index[:, 0])))
        yt = tf.math.sigmoid(tf.gather(y_true, tf.constant(index[:, 0])) - tf.gather(y_true, tf.constant(index[:, 1])))
        yp = tf.math.sigmoid(tf.gather(y_pred, tf.constant(index[:, 0])) - tf.gather(y_pred, tf.constant(index[:, 1])))
        # tf.print(tf.shape(yt))
        onetensor = tf.ones(shape=tf.shape(yt))
        # tempmatrix = (-1)*K.dot(yt, tf.math.log(tf.transpose(yp))) - K.dot((onetensor - yt),
        #                                                             tf.math.log(tf.transpose(onetensor - yp)))
        temploss = (-1)*tf.reduce_sum(tf.math.multiply(yt, tf.math.log(yp))) - tf.reduce_sum(tf.math.multiply((onetensor - yt),
                                                                    tf.math.log(onetensor - yp)))
        # tf.print(tf.shape(tempmatrix))
        # return K.mean(tf.linalg.diag_part(tempmatrix))
        return temploss
    return loss

## combine all graphs into a single graph

def combine_graphs(graphlist):
    U = nx.disjoint_union_all(graphlist)
    return U

## ===================== Generate feature vector ==========================

def get_graphnodefeatures(g):
    for node_id, node_data in g.nodes(data=True):
        node_data["feature"] = [g.degree(node_id, weight="weight"),
                                nx.average_neighbor_degree(g, nodes=[node_id], weight="weight")[node_id], 1, 1, 1]

## ===================== Generate variance of feature vector ==========================

def generate_multiple_graphinstance(g, n_mc):
    edgelist = list(g.edges())
    Listgraph = []

    for count in range(n_mc):
        print(count)
        newedgelist = []
        temprandlist = np.random.uniform(0,1, len(edgelist))
        for ind in range(len(edgelist)):
            if temprandlist[ind] <= g.edges[edgelist[ind][0], edgelist[ind][1]]['weight']:
                newedgelist.append(edgelist[ind])

        G = nx.Graph()
        G.add_edges_from(newedgelist)

        for node_id, node_data in G.nodes(data=True):
            node_data["feature"] = [G.degree(node_id, weight="weight"),
                                    nx.average_neighbor_degree(G, nodes=[node_id], weight="weight")[node_id], 1, 1, 1]

        Listgraph.append(G)

    return Listgraph

def get_feature_meanandvariance(g, listgraph):
    meandic = {}
    vardic = {}

    for node in g.nodes():
        print(node)
        templist = []
        for countg in range(len(listgraph)):
            try:
                templist.append(listgraph[countg].nodes[node]['feature'])
            except:
                pass

        # meandic[node] = np.mean(np.array(templist), axis=0)
        meandic[node] = np.sum(np.array(templist), axis=0)/len(listgraph)
        vardic[node] = np.var(np.array(templist), axis=0)

    return meandic, vardic
    ## ================ generate data frame for graph target labels ====================

def getgraphtargetdf(Listlabel, nodelist):

    # aggregate labels from all graphs
    targetlabel = Listlabel[0]
    for countlen in range(len(Listlabel) - 1):
        targetlabel = np.concatenate((targetlabel, Listlabel[countlen + 1]), axis=0)

    # gen datagrame of target labels
    targetdf = pd.DataFrame()
    targetdf['metric'] = targetlabel
    targetdf['nodename'] = nodelist
    targetdf = targetdf.set_index('nodename')

    return targetdf

## get kth order neighbors

def knbrs(G, start, k):
    nbrs = set([start])
    allnbrs = set()
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
        for val in nbrs:
            allnbrs.add(val)

    return nbrs, allnbrs


