import networkx as nx
import pandas as pd
import os
import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph import mapper
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanPoolingAggregator, MaxPoolingAggregator, MeanAggregator
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.stats as stats
from src.visualization import visualize as vs
from src.features import build_features as bf
from tensorflow.keras.models import load_model
import pickle

## ########################################### build graph ################################################
#%% ############################################################################################################

G = StellarGraph.from_networkx(g, node_features="feature")
print(G.info())

#%% #################################### Graphsage Model loadig ###########################################
#%% ############################################################################################################

batch_size = 70
num_samples = [10, 10, 5, 5]
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

targets = np.array(targetdf['btw'])

test_gen = generator.flow(targetdf.index, targets)

indices = bf.expandy(batch_size, 2)

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

## load

filepath ='.\models\\Graphsage' + fileext + '_rl.h5'

model = load_model(filepath, custom_objects={"MeanAggregator": MeanAggregator,'loss': noderankloss(indices)})
#model = load_model(filepath, custom_objects={"MeanAggregator": MeanAggregator})

##%% #################################### Model Evaluation ######################################################
#%% ############################################################################################################

all_nodes = targetdf.index
all_mapper = generator.flow(all_nodes)
y_pred = model.predict(all_mapper)

## kendall tau metric for rank
ktau, p_value = stats.kendalltau(targetdf['btw'], y_pred)

print("kendalls tau ", ktau)
print("top k perfom")

## top k pf
vs.compute_topkperf(targetdf['btw'], y_pred, 0.95)





