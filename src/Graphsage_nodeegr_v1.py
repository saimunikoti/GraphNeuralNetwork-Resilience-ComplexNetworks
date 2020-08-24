import networkx as nx
import pandas as pd
import os
from src.data.make_dataset import GenEgrData
from stellargraph_Custom import StellarGraph
from stellargraph_Custom import mapper
from stellargraph_Custom.mapper import Custom_GraphSAGENodeGenerator
from stellargraph_Custom.layer import GraphSAGE, MeanPoolingAggregator, MaxPoolingAggregator
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn import preprocessing, feature_extraction, model_selection

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.backend as K

import scipy.stats as stats
from src.visualization import visualize as vs
from src.features import build_features as bf
md = GenEgrData()

#################################### Load Dataset #########################################################

datadir = ".\data\processed\\adj_egrclass_pl_2000n_"
predictor, label, feature, listgraph = md.gen_egr_plmodel(1, 200, datadir, "adjacency", genflag=1)

######## for same graph

# listgraph.append(listgraph[0])
# label = np.concatenate((label,label), axis=1)

##
def get_nodetargets(target):
    targetdf = pd.DataFrame()
    label1 = target.flatten()
    targetdf['btw'] = label1
    return targetdf

targetdf = get_nodetargets(label)

############################################### node eature of all graphs ###############################################

def get_nodefeatures(g):
    for node_id, node_data in g.nodes(data=True):
        node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1]

############################################## build graph ###############################################
graphlist= []

for countg in range(len(listgraph)):
    get_nodefeatures(listgraph[countg])
    graphlist.append(StellarGraph.from_networkx(listgraph[countg], node_features="feature"))

train_subjects, test_subjects = model_selection.train_test_split(targetdf, train_size=0.8, test_size=None)

train_targets = np.array(train_subjects)
test_targets = np.array(test_subjects)

##%% #################################### Graphsage Model building ###########################################
#%% ############################################################################################################
""" bacth size should be a common factor of length of both training and testing data """

batch_size = 20
num_samples = [10, 5,5,10]
generator = Custom_GraphSAGENodeGenerator(graphlist, batch_size, num_samples)

train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)  # train_subjects.index for selecting training nodes
test_gen = generator.flow(test_subjects.index, test_targets)

# aggregatortype = MaxPoolingAggregator()
graphsage_model = GraphSAGE(layer_sizes=[32, 16, 8, 8], generator=generator, activations=["relu","relu","relu","linear"],
                            bias=True, dropout = 0.0)

x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="relu")(x_out)

model = Model(inputs=x_inp, outputs=prediction)

##%% ##################################### Model training #######################################################
#%% ############################################################################################################

indices = bf.expandy(batch_size, 1)

def noderankloss(index):

    def loss(y_true, y_pred):
        # tf.print(tf.gather(y_true, tf.constant(index[:, 0])))

        yt = tf.math.sigmoid(tf.gather(y_true, tf.constant(index[:, 0])) - tf.gather(y_true, tf.constant(index[:, 1])))
        yp = tf.math.sigmoid(tf.gather(y_pred, tf.constant(index[:, 0])) - tf.gather(y_pred, tf.constant(index[:, 1])))
        # tf.print(tf.shape(yt))
        onetensor = tf.ones(shape=tf.shape(yt))
        # tempmatrix = (-1)*K.dot(yt, tf.math.log(tf.transpose(yp))) - K.dot((onetensor - yt),
        #                                                tf.math.log(tf.transpose(onetensor - yp)))

        temploss = (-1)*tf.reduce_sum(tf.math.multiply(yt, tf.math.log(yp))) - tf.reduce_sum(tf.math.multiply((onetensor - yt),
                                                                    tf.math.log(onetensor - yp)))
        # tf.print(tf.shape(tempmatrix))
        # return K.mean(tf.linalg.diag_part(tempmatrix))
        return temploss
    return loss

## MODEL COMPILE AND TRAINING

model.compile( optimizer=optimizers.Adam(lr=0.005), loss = noderankloss(indices), metrics=["acc"])
# model.compile( optimizer=optimizers.Adam(lr=0.005), loss="mean_squared_error", metrics=["acc"])

filepath ='.\models\\EGR_Graphsage\\pl500-2000_m1test.h5'
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(train_gen, epochs=10, validation_data=test_gen, callbacks=[mcp], verbose=2, shuffle=False)

#%% #################################### Model Evaluation ######################################################
#%% ############################################################################################################

all_nodes = test_subjects.index
all_mapper = generator.flow(all_nodes)
y_pred = model.predict(all_mapper)

## kendall tau metric for rank
ktau, p_value = stats.kendalltau(test_targets, y_pred)
print(ktau)

## top k pf
vs.compute_topkperf(train_targets, y_pred, 0.9)

## mse error
MSE = np.mean(np.square(y_pred-train_targets), axis=0)
print(MSE)