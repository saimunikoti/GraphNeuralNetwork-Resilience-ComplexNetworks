import networkx as nx
import pandas as pd
import os
from src.data.make_dataset import GenEgrData

from stellargraph import StellarGraph
from stellargraph import mapper
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanPoolingAggregator, MaxPoolingAggregator

# from stellargraph_Custom import StellarGraph
# from stellargraph_Custom import mapper
# from stellargraph_Custom.mapper import GraphSAGENodeGenerator
# from stellargraph_Custom.layer import GraphSAGE, MeanPoolingAggregator, MaxPoolingAggregator

# import os, sys
# sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning-master"))
# sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning-master/examples")) # Adding the submodule to the module search path

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.backend as K
md=GenEgrData()
import scipy.stats as stats
from src.visualization import visualize as vs
from src.features import build_features as bf
#%% #################################### Load Dataset #########################################################
#%% ############################################################################################################
## egr data
datadir = ".\data\processed\\adj_egrclass_pl_1000n_"
#
predictor, label, feature, graph = md.gen_egr_plmodel(1, 4000, datadir, "adjacency", genflag=1)
g = graph[0]

nx.write_gpickle(g, ".\data\processed\\adj_egrclass_pl_4000n.gpickle")

## load graphs
# g = nx.read_gpickle(".\\data\processed\\Graph_2000.gpickle")

targetdf = pd.DataFrame()
label = np.reshape(label, (label.shape[1],))
targetdf['btw'] = label

## load saved egr targets of graph
# targetdf= pd.read_excel(".\\data\processed\\egr_2000.xlsx")

### feature vector
for node_id, node_data in g.nodes(data=True):
    node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1]

## ########################################### build graph ###############################################
#%% ############################################################################################################

G= StellarGraph.from_networkx(g, node_features="feature")
print(G.info())

train_subjects, test_subjects = model_selection.train_test_split(targetdf, train_size=0.8, test_size=None)

# temp_train_subjects = np.reshape(np.array(train_subjects), (train_subjects.shape[0],1))
# temp_test_subjects = np.reshape(np.array(test_subjects), (test_subjects.shape[0],1))
# train_targets = target_encoding.fit_transform(temp_train_subjects).toarray()
# test_targets = target_encoding.transform(temp_test_subjects).toarray()

train_targets = np.array(train_subjects)
test_targets = np.array(test_subjects)

#%% #################################### Graphsage Model building ###########################################
#%% ############################################################################################################

batch_size = 40
num_samples = [10, 10, 5, 5]
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)  # train_subjects.index for selecting training nodes
test_gen = generator.flow(test_subjects.index, test_targets)

# aggregatortype = MaxPoolingAggregator()
graphsage_model = GraphSAGE(layer_sizes=[32, 32, 16, 16], generator=generator, activations=["relu","relu","relu","linear"],
                            bias=True, dropout=0.0)

x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="relu")(x_out)

model = Model(inputs=x_inp, outputs=prediction)

##%% ##################################### Model training #######################################################
#%% ############################################################################################################

indices = bf.expandy(batch_size, 2)

# def cat_loss(y_true, y_pred):
#     return tf.keras.losses.categorical_crossentropy( y_true, y_pred, from_logits=False, label_smoothing=0)

# @tf.function
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

##
model.compile( optimizer=optimizers.Adam(), loss = noderankloss(indices), metrics=["acc"])
model.compile( optimizer=optimizers.Adam(lr=0.005), loss="mean_squared_error", metrics=["acc"])

filepath ='.\models\\EGR_Graphsage\\pl500-3000_m1test1.h5'
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(train_gen, epochs=15, validation_data=test_gen, callbacks=[mcp], verbose=2, shuffle=False)

#%% #################################### Model Evaluation ######################################################
#%% ############################################################################################################

all_nodes = test_subjects.index
all_mapper = generator.flow(all_nodes)
y_pred = model.predict(all_mapper)

## kendall tau metric for rank
ktau, p_value = stats.kendalltau(test_targets, y_pred)
print(ktau)

## top k pf
vs.compute_topkperf(test_targets, y_pred, 0.9)