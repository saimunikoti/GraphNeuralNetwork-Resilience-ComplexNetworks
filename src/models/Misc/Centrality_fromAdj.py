import time
import scipy.sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
import tensorflow as tf
import os
from keras import datasets, layers, models
os.chdir(r"C:\Users\saimunikoti\Manifestation\GraphLearning\GCN_Regression")

# generate data
def get_label(rankstrain):
    nodecent = np.zeros(len(rankstrain))
    for countkeys,keys in enumerate(rankstrain):
        if rankstrain[keys]>0.2:
            nodecent[countkeys]= 2
        elif rankstrain[keys]>0 and rankstrain[keys]<=0.2:
            nodecent[countkeys]= 1
        else:
            nodecent[countkeys]= 0

    return nodecent
data = []
label =[]

for countgraphs in range(2):
    g = nx.generators.random_graphs.powerlaw_cluster_graph(30,1,0.4)
    adjg = nx.adj_matrix(g)
    data.append(adjg.toarray())

    ranks = dict(nx.betweenness_centrality(g))
    templabel = get_label(ranks)
    label.append(templabel)

data = np.array(data)

## model building

model = models.Sequential()
model.add(layers.Conv2D(8, (3,), activation='relu', input_shape=(30, 30,)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(30, activation='linear'))

model.compile(optimizer='adam',
              loss='mean_absolute_percentage_error',

              metrics=['accuracy'])
history = model.fit(data, label, epochs=10,
                    validation_data=(data, label))

print(data.shape)

