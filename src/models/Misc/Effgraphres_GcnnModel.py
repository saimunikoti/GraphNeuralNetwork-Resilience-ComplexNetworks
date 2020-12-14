import networkx as nx
import numpy as np
from src.data.make_dataset import GenEgrData
from src.features import build_features as bf
from keras.models import Input, Model, Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Lambda, Maximum, AveragePooling2D, Reshape, Flatten, Average
from keras.layers import Conv2D, MaxPooling2D, Activation, Reshape
import keras.backend as K
import warnings
import tensorflow as tf
from keras.callbacks import  ModelCheckpoint, Callback
from keras import optimizers
import os, sys
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning-master"))
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning-master/examples")) # Adding the submodule to the module search path
from utils import *
from keras_dgl.layers import GraphCNN, MultiGraphCNN
from src.data import make_dataset as ut
from src.features import build_features as bf
from src.visualization import visualize as vs
from src.models import cnn_tfmodel
import scipy.stats as stats
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")
ge = GenEgrData()
# Egrdata = GenEgrData()

# Generate data

# datadir = ".\data\processed\\adj_betweenness_pl_"
# predictor, label , feature = ut.GenerateData().generate_betweenness_plmodel(10000, 50, datadir, "adjacency", genflag=0)

datadir = ".\data\processed\\adj_egrclass_ab_100_"
predictor, label, feature = ge.gen_egr_abmodel(10000, 100, datadir, "adjacency", genflag=1)
V = predictor.shape[1]

# tempfeat=[]
# for countident in range(predictor.shape[0]):
#     Idenfeat = np.identity(V)
#     tempfeat.append(np.concatenate((Idenfeat, feature[countident]), axis=1))
#
# feature = np.array(tempfeat)

xtrain, ytrain, ftrain, xtest, ytest, ftest = ge.split_data(predictor, label, feature)
xtrain, ytrain, ftrain, xval, yval, fval = ge.split_data(xtrain, ytrain, ftrain)

A, X, y = shuffle(xtrain, ftrain, ytrain)
Atest, Xtest, ytest = shuffle(xtest, ftest, ytest)
Aval, Xval, yval = shuffle(xval, fval, yval)

## filters
SYM_NORM = True  # adj = D^(-0.5)*A"*D^(-0.5), A"=A+I
num_filters = 2  # output of each input = 2*(A.shape)
graph_conv_filters = preprocess_adj_tensor_with_identity(A, SYM_NORM)
graph_conv_filters_test = preprocess_adj_tensor_with_identity(Atest, SYM_NORM)
graph_conv_filters_val = preprocess_adj_tensor_with_identity(Aval, SYM_NORM)


# model
X_input = Input(shape=(X.shape[1], X.shape[2]))
graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))

layer_gcnn1 = MultiGraphCNN(8, num_filters, activation='elu')([X_input, graph_conv_filters_input])
# layer_gcnn1 = Dropout(0.2)(layer_gcnn1)
layer_gcnn2 = MultiGraphCNN(8, num_filters, activation='elu')([layer_gcnn1, graph_conv_filters_input])
# layer_gcnn2 = Dropout(0.2)(layer_gcnn2)
layer_gcnn3 = MultiGraphCNN(8, num_filters, activation='elu')([layer_gcnn2, graph_conv_filters_input])
# layer_gcnn3 = Dropout(0.2)(layer_gcnn3)
layer_gcnn4 = Average()([layer_gcnn1, layer_gcnn2, layer_gcnn3])
# add new Graph layer with cnn
layer_gcnn4 = MultiGraphCNN(1, num_filters, activation='elu')([layer_gcnn4, graph_conv_filters_input])
# layer_gcnn3 = Dropout(0.2)(layer_gcnn3)
# layer_gcnn5 = Reshape((layer_gcnn4.shape[1]*layer_gcnn4.shape[2],))(layer_gcnn4)
layer_gcnn5 = Flatten()(layer_gcnn4)
# layer_gcnn5 = Dropout(0.2)(layer_gcnn5)
# # layer_conv5 = AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None)(layer_conv5)
layer_dense1 = Dense(V, activation='linear')(layer_gcnn5)

model = Model(inputs=[X_input, graph_conv_filters_input], outputs=layer_dense1)
model.summary()

# model = cnn_tfmodel.egrmodel2(A, X, graph_conv_filters,2)

# simple CNN regression model
# model = cnn_tfmodel.cnnmodel2(50)

## loss
indices = bf.expandy(V)
# def noderankloss_v2(index):
#
#     def loss(y_true, y_pred):
#
#         yt = tf.gather(y_true, index[:, 0], axis=1) - tf.gather(y_true, index[:, 1], axis=1)
#         yp = tf.gather(y_pred, index[:, 0], axis=1) - tf.gather(y_pred, index[:, 1], axis=1)
#
#         onetensor = tf.ones(shape=tf.shape(yt))
#
#         zerotensor = tf.zeros(shape= (tf.shape(yt)[0], tf.shape(yt)[0]))
#
#         yp_transpose = tf.transpose(yp)
#
#         part1 = K.dot(onetensor, yp_transpose)
#         part2  = K.dot(yt, yp_transpose)
#         part3 =  K.dot(onetensor, tf.math.log_sigmoid(tf.math.abs(yp_transpose)))
#
#         tempmatrix = part1 - part2 - part3
#
#         tempmatrix = tf.Print(tempmatrix, [tempmatrix, yp, y_pred, part1], "...")
#         return K.mean(tf.diag_part(tempmatrix))
#
#     return loss

# New designed metric
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]

        y_predict = np.asarray(model.predict([X_val[0], X_val[1]]))

        # y_val = np.argmax(y_val, axis=1)
        # y_predict = np.argmax(y_predict, axis=1)

        self._data.append({
            'val_rocauc': stats.kendalltau(y_val, y_predict),
        })
        return

    def get_data(self):
        return self._data

metrics = Metrics()

opt = optimizers.Adam(lr=0.001)
model.compile(loss=cnn_tfmodel.noderankloss(index=indices), optimizer=opt, metrics=['accuracy']) # optimizer minimizes loss

## compile CNN model with mean swuared error
# model.compile(loss="mean_absolute_error", optimizer='adam', metrics=['accuracy'])
nb_epochs = 250
# batch_size = int(0.9*A.shape[0])
filepath ='.\models\\EGR_GCNN\\ab100_m1.h5'
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
model.fit([X, graph_conv_filters], y, validation_data=[[Xval, graph_conv_filters_val], yval], epochs=nb_epochs, callbacks=[mcp],
          shuffle=True, verbose=1)

# fit CNN regression mode
# model.fit(xtrain, y, batch_size=batch_size, validation_split= 0.1, epochs=200, callbacks=[mcp],
#           shuffle=True, verbose=1)
#
# # load best save model
model = load_model(filepath, custom_objects={'MultiGraphCNN': MultiGraphCNN, 'loss': cnn_tfmodel.noderankloss(indices)})

# prediction

ypred = model.predict([Xtest, graph_conv_filters_test], steps=1)

## kendall tau metric for rank
ktau = vs.get_kendalltau(ytest, ypred)

## top K accuracy
Acc, f1score = vs.compute_topkperf(ytest, ypred)

### rank predictions
# ypred = vs.rank_ypred(ypred, 1.8)

# visualize results
# acc, precision, recall, f1score = vs.getacuracy(ytest, ypred)




