import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from stellargraph.core.graph import StellarGraph
from stellargraph.mapper.sampled_node_generators_bayesian import GraphSAGENodeGenerator
from stellargraph.layer.graphsage_bayesian import GraphSAGE, MaxPoolingAggregator, MeanAggregator, MeanAggregatorvariance

from tensorflow.keras import layers, optimizers, losses, metrics, Model, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import model_selection
import numpy as np
import tensorflow as tf
from src.features import build_features as bf
from src.data import config
from tensorflow.keras.models import load_model

## ======================build graph ###############################################

G = StellarGraph.from_networkx(g, node_features="feature")
print(G.info())

train_subjects, test_subjects = model_selection.train_test_split(targetdf, train_size=0.8, test_size=None)

train_targets = np.array(train_subjects)
test_targets = np.array(test_subjects)

batch_size = 40
# number of nodes to consider for each hop
num_samples = [15, 10, 5]
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)  # train_subjects.index for selecting training nodes
test_gen = generator.flow(test_subjects.index, test_targets)

# aggregatortype = MaxPoolingAggregator(),
# layer_sizes (list): Hidden feature dimensions for each layer. activations (list): Activations applied to each layer's output;therethere isther

graphsage_model = GraphSAGE(layer_sizes=[32, 32, 16], generator=generator, activations=["linear","linear","linear"],
                            bias=True, aggregator= MeanAggregator,  dropout=0.0)

x_inp, x_out = graphsage_model.in_out_tensors()

x_out = layers.Dense(units=10, activation="relu")(x_out)
x_out = layers.Dense(units=10, activation="relu")(x_out)
prediction = layers.Dense(units=train_targets.shape[1], activation="relu")(x_out)

model = Model(inputs=x_inp, outputs=prediction)
model.summary()

## %% ##################################### Model training #######################################################
#%% ############################################################################################################

indices = bf.expandy(batch_size, 2)

# def cat_loss(y_true, y_pred):
#     return tf.keras.losses.categorical_crossentropy( y_true, y_pred, from_logits=False, label_smoothing=0)

# @tf.function
def noderankloss(index):
    def loss(y_true, y_pred):
       # tf.print(tf.gather(y_true, tf.constant(index[:, 0])))

       #  index = np.array([[30,  2],
       # [57, 46],
       # [20, 23],
       # [ 7, 27],
       # [26,  2],
       # [ 6, 64],
       # [28, 27],
       # [14, 26],
       # [68, 24],
       # [66, 59],
       # [26, 56],
       # [61, 45],
       # [51, 62],
       # [66, 63],
       # [42, 36],
       # [22, 49],
       # [48, 17],
       # [67, 44],
       # [60, 59],
       # [ 2, 21],
       # [60, 32],
       # [68, 17],
       # [ 5, 51],
       # [53, 54],
       # [53, 18],
       # [13, 54],
       # [40,  5],
       # [48, 69],
       # [19,  7],
       # [58, 48],
       # [ 0, 50],
       # [67, 40],
       # [51,  3],
       # [32, 19],
       # [34, 49],
       # [45, 61],
       # [41, 19],
       # [42, 59],
       # [24, 19],
       # [50, 59],
       # [47, 17],
       # [49,  6],
       # [40,  9],
       # [53, 56],
       # [37, 11],
       # [58, 69],
       # [30, 55],
       # [59,  7],
       # [30, 31],
       # [30, 16],
       # [49,  8],
       # [68,  3],
       # [ 9, 42],
       # [21, 41],
       # [11, 35],
       # [16, 20],
       # [19, 36],
       # [14, 31],
       # [16, 52],
       # [50, 46],
       # [ 3, 38],
       # [29, 45],
       # [61, 36],
       # [53, 19],
       # [14, 16],
       # [60, 10],
       # [42,  5],
       # [23, 22],
       # [30,  4],
       # [59, 49],
       # [50, 23],
       # [38, 50],
       # [52, 36],
       # [12, 19],
       # [ 9, 51],
       # [ 6, 47],
       # [42, 11],
       # [44, 15],
       # [65, 21],
       # [ 3, 46],
       # [31, 16],
       # [28, 60],
       # [ 5,  1],
       # [57, 43],
       # [52, 24],
       # [20, 21],
       # [57,  4],
       # [49, 55],
       # [43, 52],
       # [67, 61],
       # [ 5, 11],
       # [34, 45],
       # [15, 52],
       # [44, 42],
       # [22, 13],
       # [34, 62],
       # [ 3, 21],
       # [42, 11],
       # [36,  3],
       # [ 9, 46],
       # [ 3, 29],
       # [31,  2],
       # [35, 64],
       # [62,  4],
       # [ 1, 55],
       # [29, 31],
       # [27, 52],
       # [ 9, 16],
       # [42, 49],
       # [ 7, 27],
       # [32, 27],
       # [44, 62],
       # [68, 51],
       # [39, 29],
       # [38,  4],
       # [ 6, 11],
       # [ 0, 16],
       # [15,  9],
       # [ 8, 57],
       # [34, 51],
       # [18, 24],
       # [57, 54],
       # [20, 17],
       # [23, 43],
       # [26, 37],
       # [60, 30],
       # [60, 34],
       # [40, 62],
       # [34, 61],
       # [ 4,  1],
       # [35, 32],
       # [61, 28],
       # [68, 56],
       # [ 8,  3],
       # [44, 13],
       # [ 5, 20],
       # [36, 62],
       # [34, 21]])

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
# model.compile( optimizer=optimizers.Adam(), loss = noderankloss(), metrics=["acc"])
# model.compile( optimizer=optimizers.Adam(), loss=noderankloss(indices), metrics=["acc"])
model.compile( optimizer=optimizers.Adam(), loss="mean_squared_error", metrics=["acc"], run_eagerly=True)
# model.compile( optimizer=optimizers.Adam(), loss=tf.keras.losses.Huber(delta=1.0), metrics=["acc"])
# model.compile( optimizer=optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])

filepath = config.modelpath + fileext + '_egrscoreranktest.h5'
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(train_gen, epochs=20, validation_data=test_gen, callbacks=[mcp], verbose=2, shuffle=False)

#
# model = load_model(filepath, custom_objects={"MaxPoolingAggregator": MaxPoolingAggregator,"loss": noderankloss(indices)})
model = load_model(filepath, custom_objects={"MeanAggregator": MeanAggregator})

#
# all_nodes = targetdf.index
# all_mapper = generator.flow(all_nodes)
# y_pred = model.predict(all_mapper)

## Generate results
# y_test = np.array(targetdf['metric'])
# y_prednorm = (y_pred - min(y_pred))/(np.max(y_pred)- np.min(y_pred))
#
# graphsizelist = [0, 100, 200, 400, 400, 500, 500, 800, 800, 1000, 1000, 2000, 2000, 3000, 4000]
# graphsizelist = [0, len(g.nodes)]
#
# def gen_rankresults(margin, graphsizelist, y_test, y_pred):
#
#     result = np.zeros( ((len(graphsizelist)-1), 8))
#     for countgraph in range(len(graphsizelist)-1):
#
#        temp_ytest = y_test[sum(graphsizelist[0:countgraph+1]) : sum(graphsizelist[0:countgraph+2])]
#        temp_ypred = y_pred[sum(graphsizelist[0:countgraph+1]) : sum(graphsizelist[0:countgraph+2])]
#
#        rank_test = np.array([1 if ind >= (1-margin)*np.max(temp_ytest) else 0 for ind in temp_ytest])
#        rank_pred = np.array([1 if ind >= (1-margin)*np.max(temp_ypred) else 0 for ind in temp_ypred])
#        # overall accuracy
#        result[countgraph, 0] = accuracy_score(rank_test, rank_pred)
#
#        try:
#             result[countgraph, 1] = precision_score(rank_test, rank_pred)
#             result[countgraph, 2] = recall_score(rank_test, rank_pred)
#        except:
#             print("precision not defined")
#
#        ind = np.where(rank_test == 1)[0]
#        # Top N accuracy
#        result[countgraph, 3] = sum(rank_pred[ind]) / len(ind)
#
#        rank_test = np.array([1 if ind <= margin*np.max(temp_ytest) else 0 for ind in temp_ytest])
#        rank_pred = np.array([1 if ind <= margin*np.max(temp_ypred) else 0 for ind in temp_ypred])
#
#        result[countgraph, 4] = accuracy_score(rank_test, rank_pred)
#        try:
#             result[countgraph, 5] = precision_score(rank_test, rank_pred)
#             result[countgraph, 6] = recall_score(rank_test, rank_pred)
#        except:
#            print("precision not work")
#        ind = np.where(rank_test == 1)[0]
#        result[countgraph, 7] = sum(rank_pred[ind]) / len(ind)
#
#     return result
#
# margin = 0.1
# Result = vs.gen_rankresults(margin, graphsizelist, y_test, y_pred)
# meanresult = np.mean(Result, axis=0)
#
# with open(config.modelpath + fileext + "_results_2layer_hubber.pickle", 'wb') as b:
#     pickle.dump(Result, b)

