import networkx as nx
import pandas as pd
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt

from src_bgnn.data import config as cnf

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.calibration import expected_calibration_error, plot_reliability_diagram
from stellargraph.calibration import IsotonicCalibration, TemperatureCalibration

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from stellargraph import globalvar
from stellargraph import datasets
from IPython.display import display, HTML
from tensorflow.keras.models import load_model
from stellargraph.layer import GraphSAGE, MeanPoolingAggregator, MaxPoolingAggregator, MeanAggregator
from src_bgnn.data import config as cnf
from stellargraph.layer.link_inference import LinkEmbedding
import dgl

"""
This script take raw graph
Predict link probability using graphsage link prediction model
Assign predicted link weights in edges and save networkx graph as gpickle file
Assign predicted link weights in edges, relabel nodes and store dgl graph as pickle file

"""

##
batch_size = 20
epochs = 50  # The number of training epochs for training the GraphSAGE model.

# train, test, validation split
# train_size = 0.2
# test_size = 0.15
# val_size = 0.2

## cora data
dataset = datasets.Cora()
display(HTML(dataset.description))
G, node_subjects = dataset.load(subject_as_feature=True)
node_subject = node_subjects.astype("category").cat.codes
print(G.info())

# load citeseer
dataset = datasets.CiteSeer()
display(HTML(dataset.description))

G, subjects = dataset.load(largest_connected_component_only=True)
print(G.info())
node_subject = subjects.astype("category").cat.codes

# ============== load pubmed stellargraph ================

dataset = datasets.PubMedDiabetes()
display(HTML(dataset.description))
G, _subjects = dataset.load()
print(G.info())

# ============= load amazon co purchase graph dataset from dgl========

dataset = dgl.data.AmazonCoBuyComputerDataset()

def get_nxgraph_fromdgl(data):
    gorg = data[0]
    num_class = data.num_classes
    feat = gorg.ndata['feat']  # get node feature
    label = gorg.ndata['label']  # get node labels

    gnx = gorg.to_networkx(node_attrs=['feat','label'] )
    gnx = nx.Graph(gnx)

    # load amazon features into n graph
    edgelist = list(gnx.edges)
    g = nx.Graph()
    g.add_edges_from(edgelist)

    for cnodes in g.nodes:
        g.nodes[cnodes]['feature'] = list(gnx.nodes[cnodes]['feat'].numpy())

    nodesubjects = {}

    for nodeiter in g.nodes:
        nodesubjects[nodeiter] = gnx.nodes[nodeiter]['label'].item()

    return g, nodesubjects

g, node_subject = get_nxgraph_fromdgl(dataset)

filepath = cnf.datapath + "\\amazon_computer.gpickle"
nx.write_gpickle(g, filepath)
G = StellarGraph.from_networkx(g, node_features="feature")

#========== load ppi dataset from dgl ================

dataset = dgl.data.PPIDataset()

def get_nxgraph_fromdgl(data, ind):
    gorg = data[ind]
    feat = gorg.ndata['feat']  # get node feature
    label = gorg.ndata['label']  # get node labels

    gnx = gorg.to_networkx(node_attrs=['feat','label'] )
    gnx = nx.Graph(gnx)

    # load amazon features into n graph
    edgelist = list(gnx.edges)
    g = nx.Graph()
    g.add_edges_from(edgelist)

    for cnodes in g.nodes:
        g.nodes[cnodes]['feature'] = list(gnx.nodes[cnodes]['feat'].numpy())

    nodesubjects = {}

    for nodeiter in g.nodes:
        nodesubjects[nodeiter] = gnx.nodes[nodeiter]['label'].numpy()

    return g, nodesubjects

g, node_subject = get_nxgraph_fromdgl(dataset, 13)

# convert encoded labels to integers
node_subjects = {ind: np.argmax(node_subject[ind]) for ind in node_subject.keys() }
G = StellarGraph.from_networkx(g, node_features="feature")

### Define an edge splitter on the original graph G:

edge_splitter_test = EdgeSplitter(G)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:

G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True)

# Define an edge splitter on the reduced graph G_test:
# edge_splitter_val = EdgeSplitter(G_test)
#
# # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# # reduced graph G_train with the sampled links removed:
#
# G_val, edge_ids_val, edge_labels_val = edge_splitter_val.train_test_split(
#     p=val_size, method="global", keep_connected=True
# )

# Define an edge splitter on the reduced graph G_test:

edge_splitter_train = EdgeSplitter(G)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.25, method="global", keep_connected=True
)

## get whole graph edge ids and labels

edge_splitter_test_all = EdgeSplitter(G)

G_all, edge_ids_train_all, edge_labels_train_all = edge_splitter_test_all.train_test_split(
    p=0.99999, method="global", keep_connected=False
)

edge_ids_test_all = edge_splitter_test_all.positive_edges_ids
edge_labels_test_all = edge_splitter_test_all.positive_edges_labels

##

num_samples = [20, 10]

train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

all_gen = GraphSAGELinkGenerator(G, batch_size, num_samples)
all_flow = all_gen.flow(edge_ids_test_all, edge_labels_test_all)

layer_sizes = [64, 32]

graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=train_gen, bias=True, activations=["relu","linear"], aggregator = MeanAggregator, dropout=0.2
)

# Build the model and expose input and output sockets of graphsage, for node pair inputs:
x_inp, x_out = graphsage.in_out_tensors()

# x_out = layers.Dense(units=10, activation="relu")(x_out)

logits = link_classification(
    output_dim=1, output_act="linear", edge_embedding_method="mul"
)(x_out)

prediction = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(logits)

model = tf.keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-3),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=["acc"],
)

##

filepath = cnf.modelpath + '\\ppi_linkpred.h5'

mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(train_flow, epochs = 20, validation_data=test_flow,  callbacks=[mcp], verbose=2, shuffle=False)

## prediction

from tensorflow.keras.models import load_model

model = load_model(filepath, custom_objects={"MeanAggregator": MeanAggregator, "LinkEmbedding": LinkEmbedding})

y_pred = model.predict(all_flow)

test_metrics = model.evaluate(all_flow)

print("\nTest Set Metrics of the trained model:")

for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

## save prediction as edge weights

g2 = G.to_networkx()
g3 = nx.Graph(g2)

for (n1, n2, d) in g3.edges(data=True):
    d.clear()

# assign weight 1 to edges
for u,v in g3.edges():
    g3.edges[u,v]['weight'] = 1.0

for (node1, d) in g3.nodes(data=True):
    d['label'] = node_subjects[node1]

## TODO load graph with partially weights

filepath = cnf.datapath + '\\amazon_computer_weighted' + ".gpickle"
g3 = nx.read_gpickle(filepath)

# predict edge weights
i = 0

for single_edge_id, single_edge_label in zip(edge_ids_test_all, edge_labels_test_all):
    single_edge_id = np.reshape(single_edge_id, (1, 2))
    single_edge_label = np.reshape(single_edge_label, (1,))
    t_flow = all_gen.flow(single_edge_id, single_edge_label)
    temp_pred = model.predict(t_flow)

    try:
        g3[single_edge_id[0][0]][single_edge_id[0][1]]['weight'] = temp_pred[0][0]
    except:
        print("=")
        pass

    i += 1
    print(i+1)

## some preprocessing of node labels


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(node_subjects)
nodelist_index = node_subjects.index
node_subjects_new = le.transform(node_subjects)

for count, nodeiter in enumerate(nodelist_index):
    g3.nodes[nodeiter]['label'] = node_subjects_new[count]

## save in gpickle format

filepath = cnf.datapath + '\\ppi_weighted' + ".gpickle"
nx.write_gpickle(g3, filepath, protocol=4)

## create & save graph in dgl format

import dgl
filepath = cnf.datapath + "\\citeseer_weighted.gpickle"
g3 = nx.read_gpickle(filepath)
nodelist = list(g3.nodes())
edgelist = list(g3.edges())

mapping = dict(zip(g3, range(0, len(g3.nodes()))))
# relabel nodes from 0 to ..
g3_new = nx.relabel_nodes(g3, mapping)

edgelistnew = list(g3_new.edges())

graph3 = dgl.from_networkx(g3_new, node_attrs=['feature','label'])

graph3.edata['weight'] = th.ones(graph3.num_edges(), dtype=th.float32)

for count,(u,v) in enumerate(edgelistnew):
    graph3.edata['weight'][th.tensor([count])] = th.tensor(g3_new.edges[u,v]['weight'])

import pickle
filepath = cnf.datapath + "\\citeseer_weighted.pickle"

with open(filepath, 'wb') as f:
    pickle.dump(graph3, f)

##
# ## calibration
#
# num_tests = 1
# all_test_predictions = [
#     model.predict(all_flow, verbose=True) for _ in np.arange(num_tests)
# ]
#
# calibration_data = [
#     calibration_curve(
#         y_prob=test_predictions, y_true=edge_labels_test_all, n_bins=10, normalize=True
#     )
#     for test_predictions in all_test_predictions
# ]
#
#
# for fraction_of_positives, mean_predicted_value in calibration_data:
#     ece_pre_calibration = expected_calibration_error(
#         prediction_probabilities=all_test_predictions[0],
#         accuracy=fraction_of_positives,
#         confidence=mean_predicted_value,
#     )
#     print("ECE: (before calibration) {:.4f}".format(ece_pre_calibration))
#
# ##
#
# use_platt = False
# score_model = keras.Model(inputs=x_inp, outputs=logits)
#
# if use_platt:
#     all_val_score_predictions = [
#         score_model.predict(val_flow, verbose=True) for _ in np.arange(num_tests)
#     ]
#     all_test_score_predictions = [
#         score_model.predict(test_flow, verbose=True) for _ in np.arange(num_tests)
#     ]
#     all_test_probabilistic_predictions = [
#         model.predict(test_flow, verbose=True) for _ in np.arange(num_tests)
#     ]
# else:
#     all_val_score_predictions = [
#         model.predict(val_flow, verbose=True) for _ in np.arange(num_tests)
#     ]
#     all_test_probabilistic_predictions = [
#         model.predict(all_flow, verbose=True) for _ in np.arange(num_tests)
#     ]
#
# val_predictions = np.mean(np.array(all_val_score_predictions), axis=0)
# val_predictions.shape
#
# # These are the uncalibrated prediction probabilities.
# if use_platt:
#     test_predictions = np.mean(np.array(all_test_score_predictions), axis=0)
#     test_predictions.shape
# else:
#     test_predictions = np.mean(np.array(all_test_probabilistic_predictions), axis=0)
#     test_predictions.shape
#
# if use_platt:
#     # for binary classification this class performs Platt Scaling
#     lr = TemperatureCalibration()
# else:
#     lr = IsotonicCalibration()
#
# val_predictions.shape, edge_labels_val.shape
#
# lr.fit(val_predictions, edge_labels_val)
#
# lr_test_predictions = lr.predict(test_predictions)
#
# lr_test_predictions.shape
#
# calibration_data = [
#     calibration_curve(
#         y_prob=lr_test_predictions, y_true=edge_labels_test_all, n_bins=10, normalize=True
#     )
# ]
#
# for fraction_of_positives, mean_predicted_value in calibration_data:
#     ece_post_calibration = expected_calibration_error(
#         prediction_probabilities=lr_test_predictions,
#         accuracy=fraction_of_positives,
#         confidence=mean_predicted_value,
#     )
#     print("ECE (after calibration): {:.4f}".format(ece_post_calibration))
#
# plot_reliability_diagram(
#     calibration_data, lr_test_predictions, ece=[ece_post_calibration]
# )
#
#
# y_pred = np.zeros(len(lr_test_predictions))
#
# y_pred[lr_test_predictions[:, 0] > 0.5] = 1
#
# print(
#     "Accuracy for model after calibration: {:.2f}".format(
#         accuracy_score(y_pred=y_pred, y_true=edge_labels_test_all)
#     )
# )