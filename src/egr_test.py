
import numpy as np
import networkx as nx
import pandas as pd
from src.data.make_dataset import GenEgrData
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanPoolingAggregator, MaxPoolingAggregator, MeanAggregator
from tensorflow.keras import layers, optimizers, losses, metrics, Model, models
md = GenEgrData()
import pickle
import scipy.stats as stats
from src.visualization import visualize as vs
from src.features import build_features as bf

## load test egr data

datadir = ".\data\processed\\adj_egrclass_pl_200test_"
listgraph=[]
listlabel=[]

for graphsize in range(1):
    predictor, label, feature, graphlist = md.gen_egr_plmodel(1, 5000*(graphsize+1), datadir, "adjacency", genflag=1)
    listgraph.append(graphlist[0])
    listlabel.append(label[0,:])

g = bf.combine_graphs(listgraph)

## save graph and labe;
with open(".\data\processed\\"+"5000nodes_label.pickle", 'wb') as b:
    pickle.dump(listlabel, b)

nx.write_gpickle(g, ".\data\processed\\adj_egrclass_pl_5000nodes.gpickle")

## load graph
g = nx.read_gpickle(".\\data\processed\\adj_egrclass_pl_5000nodes.gpickle")

with open(".\data\processed\\"+"5000nodes_label.pickle", 'rb') as b:
    listlabel = pickle.load(b)

##
targetlabel = listlabel[0]
for countlen in range(len(listlabel)-1):
    targetlabel = np.concatenate((targetlabel, listlabel[countlen+1]), axis=0)

##
targetdf = pd.DataFrame()
# label = np.reshape(label, (label.shape[1],))
targetdf['btw'] = targetlabel
# targetdf = targetdf.drop([99, 100])

### feature vector
for node_id, node_data in g.nodes(data=True):
    node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1]

## ############################################################################################################

G= StellarGraph.from_networkx(g, node_features="feature")
print(G.info())

test_targets = np.array(targetdf)

## #################################### Graphsage Model building ###########################################
#%% ############################################################################################################

batch_size = 40
num_samples = [10, 10, 5]
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

#%% #################################### Model Evaluation ######################################################
#%% ############################################################################################################
filepath ='.\models\\EGR_Graphsage\\pl13000_m2.h5'

model1 = models.load_model(filepath, custom_objects={"MeanAggregator": MeanAggregator})

all_nodes = targetdf.index.values
all_mapper = generator.flow(all_nodes)
y_pred = model1.predict(all_mapper)

## kendall tau metric for rank
ktau, p_value = stats.kendalltau(targetdf['btw'], y_pred)
print(ktau)

## top k pf
vs.compute_topkperf(targetdf['btw'], y_pred, 0.9)
