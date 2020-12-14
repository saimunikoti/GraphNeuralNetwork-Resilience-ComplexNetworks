import networkx as nx
import numpy as np
from src.data.make_dataset import GenEgrData
from src.features import build_features as bf
from keras.models import Input, Model, Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Lambda, Maximum, AveragePooling2D, Flatten, Average
from keras.layers import Conv2D, MaxPooling2D, Activation, Reshape
import keras.backend as K
from keras.regularizers import l2
import warnings
import tensorflow as tf
from keras.callbacks import  ModelCheckpoint, Callback
from keras import optimizers
import os, sys
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning-master"))
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning-master/examples")) # Adding the submodule to the module search path
from utils import *
from keras_dgl.layers import MultiGraphAttentionCNN
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

datadir = ".\data\processed\\adj_egrclass_pl_"
predictor, label, feature = ge.gen_egr_plmodel(10000, 50, datadir, "adjacency", genflag=0)

xtrain, ytrain, ftrain, xtest, ytest, ftest = ge.split_data(predictor, label, feature)
xtrain, ytrain, ftrain, xval, yval, fval = ge.split_data(xtrain, ytrain, ftrain)

A, X, y = shuffle(xtrain, ftrain, ytrain)
Atest, Xtest, ytest = shuffle(xtest, ftest, ytest)
Aval, Xval, yval = shuffle(xval, fval, yval)
num_graph_nodes = A.shape[1]
# num_graphs = int(A.shape[0] / A.shape[1])
V = ytest.shape[1]
## filters
SYM_NORM = True  # adj = D^(-0.5)*A"*D^(-0.5), A"=A+I
num_filters = 2  # output of each input = 2*(A.shape)
graph_conv_filters = preprocess_adj_tensor_with_identity(A, SYM_NORM)
graph_conv_filters_test = preprocess_adj_tensor_with_identity(Atest, SYM_NORM)
graph_conv_filters_val = preprocess_adj_tensor_with_identity(Aval, SYM_NORM)

### set daigonal values to 1 in adjacency matrices
def getdiagonal_one(amat):
    A_eye_tensor = []
    for _ in range(amat.shape[0]):
        Identity_matrix = np.eye(V)
        A_eye_tensor.append(Identity_matrix)

    A_eye_tensor = np.array(A_eye_tensor)
    amat = np.add(amat, A_eye_tensor)
    return amat
A = getdiagonal_one(A)
Aval = getdiagonal_one(Aval)
Atest = getdiagonal_one(Atest)

### model
X_input = Input(shape=(X.shape[1], X.shape[2]))
A_input = Input(shape=(A.shape[1], A.shape[2]))

graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))

layer_gcnn1 = MultiGraphAttentionCNN(8, num_filters=num_filters, num_attention_heads=2, attention_combine='concat', name="layer_gcnn1",attention_dropout=0.5, activation='elu', kernel_regularizer=l2(5e-4))([X_input, A_input, graph_conv_filters_input])
layer_gcnn1 = Dropout(0.2)(layer_gcnn1)
layer_gcnn2 = MultiGraphAttentionCNN(8, num_filters=num_filters, num_attention_heads=2, attention_combine='concat', name="layer_gcnn2",attention_dropout=0.5, activation='elu', kernel_regularizer=l2(5e-4))([layer_gcnn1, A_input, graph_conv_filters_input])
layer_gcnn2 = Dropout(0.2)(layer_gcnn2)
# layer_gcnn3 = MultiGraphAttentionCNN(8, num_filters=num_filters, num_attention_heads=1, attention_combine='average', attention_dropout=0.5, activation='elu', kernel_regularizer=l2(5e-4))([layer_gcnn2, A_input, graph_conv_filters_input])
# layer_gcnn3 = Lambda(lambda x: K.mean(x, axis=1))(layer_gcnn2)
layer_gcnn4 = Average()([layer_gcnn1, layer_gcnn2])
#
layer_gcnn4 = MultiGraphAttentionCNN(3, num_filters=num_filters, num_attention_heads=1, attention_combine='average',name="layer_gcnn3", attention_dropout=0.5, activation='elu', kernel_regularizer=l2(5e-4))([layer_gcnn4, A_input, graph_conv_filters_input])
layer_gcnn4 = Dropout(0.2)(layer_gcnn4)
# layer_gcnn5 = Reshape((layer_gcnn4.shape[1]*layer_gcnn4.shape[2],))(layer_gcnn4)
layer_gcnn5 = Flatten()(layer_gcnn4)
layer_dense1 = Dense(V, activation='linear',name="layer_dense1")(layer_gcnn5)

model = Model(inputs=[X_input, A_input, graph_conv_filters_input], outputs=layer_dense1)
model.summary()

def visualize_conv_layer(layer_name):
    layer_output = model.get_layer(layer_name).output

    intermediate_model = Model(inputs=model.input, outputs=layer_output)

    intermediate_prediction = intermediate_model.predict([[X, A, graph_conv_filters]])

    print(intermediate_prediction.shape)

visualize_conv_layer("layer_gcnn1")
# model = cnn_tfmodel.egrmodel2(A, X, graph_conv_filters,2)

# simple CNN regression model
# model = cnn_tfmodel.cnnmodel2(50)

## loss
indices = bf.expandy(V)
# checking loss functions for node ranking
def noderankloss(index):

    def loss(y_true, y_pred):
        # index = expandy(50)
        # yt = y_true[:,index[:, 0]] - y_true[:, index[:, 1]]
        yt = tf.math.sigmoid(tf.gather(y_true, index[:, 0], axis=1) - tf.gather(y_true, index[:, 1], axis=1))
        yp = tf.math.sigmoid(tf.gather(y_pred, index[:, 0], axis=1) - tf.gather(y_pred, index[:, 1], axis=1))
        # yt = y_true[:,index[:, 0]] - y_true[:, index[:, 1]]
        # yp = y_pred[:,index[:, 0]] - y_pred[:, index[:, 1]]
        # yp = tf.Print(yp, ["ypred", y_pred[1,:]])
        onetensor = tf.ones(shape=tf.shape(yt))
        tempmatrix = (-1)*K.dot(yt, K.log(tf.transpose(yp))) - K.dot((onetensor - yt),
                                                                K.log(tf.transpose(onetensor - yp)))
        # tempmatrix = tf.Print(tempmatrix, [yp], "...")
        return K.mean(tf.diag_part(tempmatrix))

    return loss

# class Metrics(Callback):
#     def on_train_begin(self, logs={}):
#         self._data = []
#
#     def on_epoch_end(self, batch, logs={}):
#         X_val, y_val = self.validation_data[0], self.validation_data[1]
#
#         y_predict = np.asarray(model.predict([X_val[0], X_val[1]]))
#
#         # y_val = np.argmax(y_val, axis=1)
#         # y_predict = np.argmax(y_predict, axis=1)
#
#         self._data.append({
#             'val_rocauc': stats.kendalltau(y_val, y_predict),
#         })
#         return
#
#     def get_data(self):
#         return self._data

# metrics = Metrics()

opt = optimizers.Adam(lr=0.001)
model.compile(loss=noderankloss(index=indices), optimizer=opt, metrics=['accuracy']) # optimizer minimizes loss

## compile CNN model with mean swuared error
# model.compile(loss="mean_absolute_error", optimizer='adam', metrics=['accuracy'])

nb_epochs = 1
filepath ='.\models\\EGR_GCNNAtt\\pl50_m1.h5'
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
model.fit([X, A, graph_conv_filters], y, validation_data=[[Xval, Aval, graph_conv_filters_val], yval], epochs=nb_epochs,
          callbacks=[mcp], shuffle=True, verbose=1)


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




