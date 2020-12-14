import time
import scipy.sparse
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
import tensorflow as tf
import os
os.chdir(r"C:\Users\saimunikoti\Manifestation\centrality_learning")
import layers.graph as lg
from src.data import make_dataset as ut
from sklearn.ensemble import RandomForestRegressor

# generate data
newdata =[]
data, label = ut.GenerateData().generate_betweennessdata(30000, genflag=0)  # flag=0 for loading saved data
newdata = np.array([data[ind].flatten() for ind in range(data.shape[0])])

xtrain, ytrain, xval, yval, xtest, ytest = ut.GenerateData().split_data(newdata, label)

# xgboo
regr = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=140, criterion='mse')
regr.fit(xtrain, ytrain)

ypred = regr.predict(xtest)

regr.score(xval, yval)
