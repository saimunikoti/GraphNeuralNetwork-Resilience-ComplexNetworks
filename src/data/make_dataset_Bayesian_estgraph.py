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
filepath = config.datapath + "Bayesian\\" + "Wmlesupervised_2000plc.mat"

W = io.loadmat(filepath)
W = W['Wpred']
W = W.toarray()

Listlabel = []
gest = nx.from_numpy_matrix(W)

Listlabel.append(ut.get_estgraphlabel(gest, "egr"))

fileext = "\\plc_2000_egr_estsupbayesian"

with open(config.datapath + 'Bayesian'+ fileext + "_label.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

## ================ load label ==============
with open(config.datapath + 'Bayesian' + fileext+ "_label.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

## === generate target vector by combniing labels from all graphs and keep it in target data frame ============

targetdf = bf.getgraphtargetdf(Listlabel, gest)
targetdf.loc[targetdf.metric==0,'metric'] = 0.001

category = pd.cut(targetdf.metric, bins=[0,0.25,0.7,1.0],labels=[0, 1, 2])
targetdf['metric'] = category
plt.hist(targetdf['metric'])
targetdf = pd.get_dummies(targetdf.metric)

## ====================== assign feature vector to each graph node ==========================

bf.get_graphnodefeatures(gest)

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

