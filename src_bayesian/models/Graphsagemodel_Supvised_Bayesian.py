import tqdm
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanPoolingAggregator, MaxPoolingAggregator, MeanAggregator, AttentionalAggregator
from stellargraph.layer import GraphSAGE
import stellargraph as sg
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, optimizers, losses, metrics, Model, models
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn import preprocessing, feature_extraction, model_selection
import numpy as np
import tensorflow as tf
import scipy.stats as stats
from src.visualization import visualize as vs
from src.features import build_features as bf

## ########################################### build graph ###############################################

G = StellarGraph.from_networkx(gest, node_features="feature")
print(G.info())

train_subjects, test_subjects = model_selection.train_test_split(targetdf, train_size=0.8, test_size=None)

train_targets = np.array(train_subjects)
test_targets = np.array(test_subjects)
##
def get_dropout(input_tensor, p=0.1, mc=False):
   if mc:
      return Dropout(p)(input_tensor, training=True)
   else:
      return Dropout(p)(input_tensor)

## ======================== Graphsage Model building ===========================

batch_size = 40
# number of nodes to consider for each hop
num_samples = [15, 10, 5]
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)  # train_subjects.index for selecting training nodes
test_gen = generator.flow(test_subjects.index, test_targets)

graphsage_model = GraphSAGE(layer_sizes=[64, 32, 16], generator=generator, aggregator = MaxPoolingAggregator,  activations=["relu","relu","linear"],
                            bias=True, dropout=0.1)

x_inp, x_out = graphsage_model.in_out_tensors()
x_out = get_dropout(x_out, p=0.1, mc='mc')
prediction = layers.Dense(units=train_targets.shape[1], activation="relu")(x_out)

model_bayesian = Model(inputs=x_inp, outputs=prediction)

##%% ##################################### Model training #######################################################

indices = bf.expandy(batch_size, 2)

# def cat_loss(y_true, y_pred):
#     return tf.keras.losses.categorical_crossentropy( y_true, y_pred, from_logits=False, label_smoothing=0)

# @tf.function
def noderankloss():

    def loss(y_true, y_pred):
        # tf.print(tf.gather(y_true, tf.constant(index[:, 0])))

        index = np.array([[30,  2],
       [57, 46],
       [20, 23],
       [ 7, 27],
       [26,  2],
       [ 6, 64],
       [28, 27],
       [14, 26],
       [68, 24],
       [66, 59],
       [26, 56],
       [61, 45],
       [51, 62],
       [66, 63],
       [42, 36],
       [22, 49],
       [48, 17],
       [67, 44],
       [60, 59],
       [ 2, 21],
       [60, 32],
       [68, 17],
       [ 5, 51],
       [53, 54],
       [53, 18],
       [13, 54],
       [40,  5],
       [48, 69],
       [19,  7],
       [58, 48],
       [ 0, 50],
       [67, 40],
       [51,  3],
       [32, 19],
       [34, 49],
       [45, 61],
       [41, 19],
       [42, 59],
       [24, 19],
       [50, 59],
       [47, 17],
       [49,  6],
       [40,  9],
       [53, 56],
       [37, 11],
       [58, 69],
       [30, 55],
       [59,  7],
       [30, 31],
       [30, 16],
       [49,  8],
       [68,  3],
       [ 9, 42],
       [21, 41],
       [11, 35],
       [16, 20],
       [19, 36],
       [14, 31],
       [16, 52],
       [50, 46],
       [ 3, 38],
       [29, 45],
       [61, 36],
       [53, 19],
       [14, 16],
       [60, 10],
       [42,  5],
       [23, 22],
       [30,  4],
       [59, 49],
       [50, 23],
       [38, 50],
       [52, 36],
       [12, 19],
       [ 9, 51],
       [ 6, 47],
       [42, 11],
       [44, 15],
       [65, 21],
       [ 3, 46],
       [31, 16],
       [28, 60],
       [ 5,  1],
       [57, 43],
       [52, 24],
       [20, 21],
       [57,  4],
       [49, 55],
       [43, 52],
       [67, 61],
       [ 5, 11],
       [34, 45],
       [15, 52],
       [44, 42],
       [22, 13],
       [34, 62],
       [ 3, 21],
       [42, 11],
       [36,  3],
       [ 9, 46],
       [ 3, 29],
       [31,  2],
       [35, 64],
       [62,  4],
       [ 1, 55],
       [29, 31],
       [27, 52],
       [ 9, 16],
       [42, 49],
       [ 7, 27],
       [32, 27],
       [44, 62],
       [68, 51],
       [39, 29],
       [38,  4],
       [ 6, 11],
       [ 0, 16],
       [15,  9],
       [ 8, 57],
       [34, 51],
       [18, 24],
       [57, 54],
       [20, 17],
       [23, 43],
       [26, 37],
       [60, 30],
       [60, 34],
       [40, 62],
       [34, 61],
       [ 4,  1],
       [35, 32],
       [61, 28],
       [68, 56],
       [ 8,  3],
       [44, 13],
       [ 5, 20],
       [36, 62],
       [34, 21]])

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
model_bayesian.compile( optimizer=optimizers.Adam(), loss = noderankloss(), metrics=["acc"])
model_bayesian.compile( optimizer=optimizers.Adam(), loss="mean_squared_error", metrics=["acc"])

filepath ='.\models\\BayesianGraphsage' + fileext + '_rankloss_3layers.h5'
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')

history = model_bayesian.fit(train_gen, epochs=30, validation_data=test_gen, callbacks=[mcp], verbose=1, shuffle=False)

sg.utils.plot_history(history)

## load model

filepath ='.\models\\BayesianGraphsage' + fileext + '_mse_3layers.h5'
model = load_model(filepath, custom_objects={"MaxPoolingAggregator": MaxPoolingAggregator})

all_nodes = targetdf.index
all_mapper = generator.flow(all_nodes)
y_pred = model.predict(all_mapper)

mc_predictions = []
for i in tqdm.tqdm(range(100)):
    y_p = model_bayesian.predict(all_mapper)
    mc_predictions.append(y_p)

def predict_dist(X, model, num_samples):
   preds = [model(X, training=True) for _ in range(num_samples)]
   return np.hstack(preds)

def predict_point(X, model, num_samples):
   pred_dist = predict_dist(X, model, num_samples)
   return pred_dist.mean(axis=1)

mc_predictions = np.array(mc_predictions)

mc_ensemble_pred = (mc_predictions).mean(axis=0)

def get_ranksfrommetic( mc_predictions, focusnode):
   listnoderanks=[]

   for count in range(mc_predictions.shape[0]):
      temp = np.argsort(mc_predictions[count], axis=0)
      listnoderanks.append(np.where(temp==focusnode)[0])

   return np.array(listnoderanks)

top10noderankspred = get_ranksfrommetic(mc_predictions,10)
plt.hist(top10noderankspred)

ktau, p_value = stats.kendalltau(targetdf['metric'], mc_ensemble_pred)

vs.compute_topkperf(targetdf['metric'], mc_ensemble_pred, 0.95)
