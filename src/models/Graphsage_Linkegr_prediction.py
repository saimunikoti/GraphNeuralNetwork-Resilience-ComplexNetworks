import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from stellargraph import StellarGraph
from stellargraph.layer import GraphSAGE, MeanPoolingAggregator, MaxPoolingAggregator,AttentionalAggregator, MeanAggregator, AttentionalAggregator
from tensorflow.keras import layers, optimizers, losses, metrics, Model, models
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

import numpy as np
import tensorflow as tf

from src.visualization import visualize as vs
from src.features import build_features as bf
import pickle
from src.data import config
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE

## ======================build graph ###############################################

train_indices = np.random.choice(len(g.edges), int(0.8*len(g.edges)), replace=False)
test_indices = np.array(list(set(np.arange(0,len(g.edges))) - set(train_indices)))

link_ids_train = link_ids[train_indices]
link_ids_test = link_ids[test_indices]

link_labels_train = link_labels[train_indices]
link_labels_test = link_labels[test_indices]

g_train = g.copy()
edgelist = [(start, end) for start, end in zip(link_ids_test[:,0], link_ids_test[:,1]) ]
g_train.remove_edges_from(edgelist)

g_test = g.copy()
edgelist = [(start, end) for start, end in zip(link_ids_train[:,0], link_ids_train[:,1]) ]
g_test.remove_edges_from(edgelist)

G = StellarGraph.from_networkx(g, node_features="feature")
g_train = StellarGraph.from_networkx(g_train, node_features="feature")
g_test = StellarGraph.from_networkx(g_test, node_features="feature")

print(g_train.info())
print(g_test.info())

##
batch_size = 40
num_samples = [15, 10, 5]

train_gen = GraphSAGELinkGenerator(g_train, batch_size, num_samples)
train_flow = train_gen.flow(link_ids_train, link_labels_train, shuffle=True)

test_gen = GraphSAGELinkGenerator(g_test, batch_size, num_samples)
test_flow = test_gen.flow(link_ids_test, link_labels_test)

traintest_gen = GraphSAGELinkGenerator(G, batch_size, num_samples)
traintest_flow = traintest_gen.flow(link_ids, link_labels)

## =================== model -==============================
graphsage_model = GraphSAGE(layer_sizes=[64, 32, 16], generator= train_gen, activations=["relu","relu","linear"],
                            bias=True, aggregator = MaxPoolingAggregator, dropout=0.0)

x_inp, x_out = graphsage_model.in_out_tensors()

def custom_layer(x):
    if isinstance(x, (list, tuple)):
        if len(x) != 2:
            raise ValueError("Expecting a list of length 2 for link embedding")
        x0 = x[0]
        x1 = x[1]
    elif isinstance(x, tf.Tensor):
        if int(x.shape[-2]) != 2:
            raise ValueError(
                "Expecting a tensor of shape 2 along specified axis for link embedding"
            )
        x0, x1 = tf.unstack(x, axis=-2)
    else:
        raise TypeError("Expected a list, tuple, or Tensor as input")

    out = tf.multiply(x0, x1)
    out = tf.keras.activations.get("linear")(out)

    return out

lambda_layer = tf.keras.layers.Lambda(custom_layer, name="lambda_layer")(x_out)

# prediction = link_regression(
#     output_dim=16, edge_embedding_method="avg"
# )(lambda_layer)

prediction = layers.Dense(units=12, activation="relu")(lambda_layer)
prediction = layers.Dense(units=8, activation="relu")(prediction)
prediction = layers.Dense(units=1, activation="relu")(prediction)

model = Model(inputs=x_inp, outputs=prediction)
model.summary()

##====================================== Model training =====================================

# indices = bf.expandy(batch_size, 2)
# # @tf.function
# def noderankloss(index):
#     def loss(y_true, y_pred):
#        # tf.print(tf.gather(y_true, tf.constant(index[:, 0])))
#
#        #  index = np.array([[30,  2],
#        # [57, 46],
#        # [20, 23],
#        # [ 7, 27],
#        # [26,  2],
#        # [ 6, 64],
#        # [28, 27],
#        # [14, 26],
#        # [68, 24],
#        # [66, 59],
#        # [26, 56],
#        # [61, 45],
#        # [51, 62],
#        # [66, 63],
#        # [42, 36],
#        # [22, 49],
#        # [48, 17],
#        # [67, 44],
#        # [60, 59],
#        # [ 2, 21],
#        # [60, 32],
#        # [68, 17],
#        # [ 5, 51],
#        # [53, 54],
#        # [53, 18],
#        # [13, 54],
#        # [40,  5],
#        # [48, 69],
#        # [19,  7],
#        # [58, 48],
#        # [ 0, 50],
#        # [67, 40],
#        # [51,  3],
#        # [32, 19],
#        # [34, 49],
#        # [45, 61],
#        # [41, 19],
#        # [42, 59],
#        # [24, 19],
#        # [50, 59],
#        # [47, 17],
#        # [49,  6],
#        # [40,  9],
#        # [53, 56],
#        # [37, 11],
#        # [58, 69],
#        # [30, 55],
#        # [59,  7],
#        # [30, 31],
#        # [30, 16],
#        # [49,  8],
#        # [68,  3],
#        # [ 9, 42],
#        # [21, 41],
#        # [11, 35],
#        # [16, 20],
#        # [19, 36],
#        # [14, 31],
#        # [16, 52],
#        # [50, 46],
#        # [ 3, 38],
#        # [29, 45],
#        # [61, 36],
#        # [53, 19],
#        # [14, 16],
#        # [60, 10],
#        # [42,  5],
#        # [23, 22],
#        # [30,  4],
#        # [59, 49],
#        # [50, 23],
#        # [38, 50],
#        # [52, 36],
#        # [12, 19],
#        # [ 9, 51],
#        # [ 6, 47],
#        # [42, 11],
#        # [44, 15],
#        # [65, 21],
#        # [ 3, 46],
#        # [31, 16],
#        # [28, 60],
#        # [ 5,  1],
#        # [57, 43],
#        # [52, 24],
#        # [20, 21],
#        # [57,  4],
#        # [49, 55],
#        # [43, 52],
#        # [67, 61],
#        # [ 5, 11],
#        # [34, 45],
#        # [15, 52],
#        # [44, 42],
#        # [22, 13],
#        # [34, 62],
#        # [ 3, 21],
#        # [42, 11],
#        # [36,  3],
#        # [ 9, 46],
#        # [ 3, 29],
#        # [31,  2],
#        # [35, 64],
#        # [62,  4],
#        # [ 1, 55],
#        # [29, 31],
#        # [27, 52],
#        # [ 9, 16],
#        # [42, 49],
#        # [ 7, 27],
#        # [32, 27],
#        # [44, 62],
#        # [68, 51],
#        # [39, 29],
#        # [38,  4],
#        # [ 6, 11],
#        # [ 0, 16],
#        # [15,  9],
#        # [ 8, 57],
#        # [34, 51],
#        # [18, 24],
#        # [57, 54],
#        # [20, 17],
#        # [23, 43],
#        # [26, 37],
#        # [60, 30],
#        # [60, 34],
#        # [40, 62],
#        # [34, 61],
#        # [ 4,  1],
#        # [35, 32],
#        # [61, 28],
#        # [68, 56],
#        # [ 8,  3],
#        # [44, 13],
#        # [ 5, 20],
#        # [36, 62],
#        # [34, 21]])
#
#         yt = tf.math.sigmoid(tf.gather(y_true, tf.constant(index[:, 0])) - tf.gather(y_true, tf.constant(index[:, 1])))
#         yp = tf.math.sigmoid(tf.gather(y_pred, tf.constant(index[:, 0])) - tf.gather(y_pred, tf.constant(index[:, 1])))
#         # tf.print(tf.shape(yt))
#         onetensor = tf.ones(shape=tf.shape(yt))
#         # tempmatrix = (-1)*K.dot(yt, tf.math.log(tf.transpose(yp))) - K.dot((onetensor - yt),
#         #                                                             tf.math.log(tf.transpose(onetensor - yp)))
#         temploss = (-1)*tf.reduce_sum(tf.math.multiply(yt, tf.math.log(yp))) - tf.reduce_sum(tf.math.multiply((onetensor - yt),
#                                                                     tf.math.log(onetensor - yp)))
#         # tf.print(tf.shape(tempmatrix))
#         # return K.mean(tf.linalg.diag_part(tempmatrix))
#         return temploss
#     return loss

##

model.compile( optimizer=optimizers.Adam(), loss="mean_squared_error", metrics=["acc"])
# model.compile( optimizer=optimizers.Adam(), loss = noderankloss(indices), metrics=["acc"])

filepath = config.linkmodel + fileext + '_scorerank.h5'

mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(train_flow, epochs = 30, validation_data= test_flow, callbacks=[mcp], verbose=2, shuffle=False)

from tensorflow.keras.models import load_model

model = load_model(filepath, custom_objects={"MaxPoolingAggregator": MaxPoolingAggregator,"custom_layer":custom_layer})

init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")

for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

y_pred = model.predict(test_flow)
y_pred = model.predict(traintest_flow)

## Generate results

y_test = np.array(link_labels)

y_prednorm = (y_pred - min(y_pred))/(np.max(y_pred)- np.min(y_pred))

graphsizelist = [0, 100, 200, 400, 400, 500, 500, 800, 800, 1000, 1000, 2000, 2000, 3000, 4000]
graphsizelist = [0, len(g.nodes)]

def gen_rankresults(margin, graphsizelist, y_test, y_pred):

    result = np.zeros( ((len(graphsizelist)-1), 8))

    for countgraph in range(len(graphsizelist)-1):

       temp_ytest = y_test[sum(graphsizelist[0:countgraph+1]) : sum(graphsizelist[0:countgraph+2])]
       temp_ypred = y_pred[sum(graphsizelist[0:countgraph+1]) : sum(graphsizelist[0:countgraph+2])]

       rank_test = np.array([1 if ind >= (1-margin)*np.max(temp_ytest) else 0 for ind in temp_ytest])
       rank_pred = np.array([1 if ind >= (1-margin)*np.max(temp_ypred) else 0 for ind in temp_ypred])

       # overall accuracy
       result[countgraph, 0] = accuracy_score(rank_test, rank_pred)

       try:
            result[countgraph, 1] = precision_score(rank_test, rank_pred)
            result[countgraph, 2] = recall_score(rank_test, rank_pred)
       except:
            print("precision not defined")

       ind = np.where(rank_test == 1)[0]

       # Top N accuracy
       result[countgraph, 3] = sum(rank_pred[ind]) / len(ind)

       rank_test = np.array([1 if ind <= margin*np.max(temp_ytest) else 0 for ind in temp_ytest])
       rank_pred = np.array([1 if ind <= margin*np.max(temp_ypred) else 0 for ind in temp_ypred])

       result[countgraph, 4] = accuracy_score(rank_test, rank_pred)
       try:
            result[countgraph, 5] = precision_score(rank_test, rank_pred)
            result[countgraph, 6] = recall_score(rank_test, rank_pred)
       except:
           print("precision not work")
       ind = np.where(rank_test == 1)[0]
       result[countgraph, 7] = sum(rank_pred[ind]) / len(ind)

    return result

margin = 0.1
rank_test = np.array([1 if ind >= (1 - margin) * np.max(y_test) else 0 for ind in y_test])
rank_pred = np.array([1 if ind >= (1 - margin) * np.max(y_pred) else 0 for ind in y_pred])

rank_test = np.array([1 if ind <= margin*np.max(y_test) else 0 for ind in y_test])
rank_pred = np.array([1 if ind <= margin*np.max(y_pred) else 0 for ind in y_pred])

accuracy_score(rank_test, rank_pred)
recall_score(rank_test, rank_pred)
precision_score(rank_test, rank_pred)

Result = vs.gen_rankresults(margin, graphsizelist, y_test, y_pred)
meanresult = np.mean(Result, axis=0)

with open(config.modelpath + fileext + "_results_2layer_hubber.pickle", 'wb') as b:
    pickle.dump(Result, b)


