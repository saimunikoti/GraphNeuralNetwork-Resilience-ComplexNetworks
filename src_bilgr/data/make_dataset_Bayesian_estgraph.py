import networkx as nx
from src.data import config
from src.data import utils as ut
from src.features import build_features as bf
import scipy.io as io
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ============ load estimated graphs weighted adjacency matrix from matlab ===============
filepath = config.datapath + "Bayesian\\" + "WmleSupervised_5000plc.mat"
Wpred = io.loadmat(filepath)
Wpred = Wpred['Wpred']
Wpred = Wpred.toarray()

# filepath = r"C:\Users\saimunikoti\Manifestation\centrality_learning\data\Bayesian\plc_5000_gobsnoise_bayesian_wadjacency.mat"
#
# W = io.loadmat(filepath)
# W = W['wadjacency']

## normalize each row
# Wprednorm = Wpred.copy()
# Wprednorm  = Wprednorm*1000
#
# for countrow in range(Wpred.shape[0]):
#     amin = min(Wpred[countrow,:])
#     amax = max(Wpred[countrow,:])
#     temp = (Wpred[countrow,:] - amin)
#     temp1 = temp/(amax-amin)
#     Wprednorm[countrow,:] = temp1
#
# # remoe Nan from the adjacenency matrix weighted
# temp = np.isnan(Wprednorm).any(axis=1)
# tempt = np.where(temp==True)[0]
#
# Wprednorm = np.delete(Wprednorm, tempt, 0)
# Wprednorm = np.delete(Wprednorm, tempt, 1)

gest = nx.from_numpy_matrix(Wpred)

# Listlabel = []
# Listlabel.append(ut.get_estgraphlabel(gest, "egr", weightflag=0))

# remove specific columns from label
# Labelarray = Listlabel[0]
# Labelarraynew = np.delete(Labelarray, [0])
# Labelarraynew = np.delete(Labelarraynew, tempt)

fileext = "\\plc_5000_egr_estsupbayesian"
nx.write_gpickle(gest, config.datapath + 'Bayesian'+ fileext + ".gpickle")

# with open(config.datapath + 'Bayesian'+ fileext + "_label.pickle", 'wb') as b:
#     pickle.dump(Listlabel, b)

## ================ load estimated graph and gobs label ==============
fileext = "\\plc_5000_egr_estsupbayesian"
gest = nx.read_gpickle(config.datapath + 'Bayesian'+ fileext+".gpickle")

fileext = "\\plc_5000_egr_bayesian"

with open(config.datapath + 'Bayesian' + fileext+ "_label.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

# remove enties for which no node is present
Labelarray = Listlabel[0]
Labelarray = np.delete(Labelarray, [0])

## ================ load noisy graph and gobs label ==============

fileext = "\\plc_5000_gobsnoiseadd_bayesian_6195"

gobsnoise = nx.read_gpickle(config.datapath + 'Bayesian'+ fileext+".gpickle")
print("", len(gobsnoise.edges))
print("", len(gobsnoise.nodes))

fileext = "\\plc_5000_egr_bayesian"

with open(config.datapath + 'Bayesian' + fileext+ "_label.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

# remove enties for which no node is present
Labelarray = Listlabel[0]
Labelarray = np.delete(Labelarray, [0])

gobsnoise.remove_node(0)

## === generate target vector by combniing labels from all graphs and keep it in target data frame ============

nodelist = list(gobsnoise.nodes)
nodelist = list(gest.nodes)
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
targetdf = pd.get_dummies(targetdf.metric)

## ====================== assign feature vector to each graph node ==========================
bf.get_graphnodefeatures(gest)
bf.get_graphnodefeatures(gobsnoise)

## ============= save weighted adjacency matrix ===================

W = nx.adjacency_matrix(g, weight='weight')
W = W.toarray()

A = nx.adjacency_matrix(g, weight=None)
A = A.toarray()

tempdic = {}
tempdic['wadjacency'] = W
tempdic['adjacency'] = A
filename = config.datapath + "Bayesian"+ fileext+ "_wadjacency.mat"
io.savemat(filename, tempdic)

