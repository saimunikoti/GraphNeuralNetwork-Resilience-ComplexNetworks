from keras.layers import Dense, Activation, Dropout, Input, Average, Maximum, AveragePooling2D, Reshape, Flatten
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from src.data.make_dataset import GenerateData
from keras.models import Input, Model, Sequential, load_model
from keras.callbacks import  ModelCheckpoint
from keras.layers import Dense, Activation, Dropout, Lambda, Maximum, AveragePooling2D, Reshape, Flatten, Average
import os, sys
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning-master"))
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning-master/examples")) # Adding the submodule to the module search path
from utils import *
from keras_dgl.layers import GraphCNN, MultiGraphCNN
from src.visualization import visualize as vs
from sklearn.utils import shuffle
from keras.utils import to_categorical

# print("Creating our simple sample data...")
# A = np.array([[0,1,5], [1,0,0], [5,0,0]])
# print(A)
# X = np.array([[1,2,10], [4,3,10], [0,2,11]]) # features, whatever we have there...
#
# # Notice, if we set A = identity matrix, then we'd effectively assume no edges and just do a basic
# # MLP on the features.
#
# # We could do the same by setting the graph_conv_filter below to Id.
#
# # We could also set X to Id, and thus effectively assume no features, and in this way
# # do an "edge" embedding, so effectively try to understand what's connected to what.
#
# # We could then use that as feature in any way we like...
#
# Y_o_dim = np.array([1,2,1])
# Y =  to_categorical(Y_o_dim) # labels, whatever we wanna classify things into... in categorical form.
# train_on_weight= np.array([1,1,0])
# print("Now we won't do any fancy preprocessing, just basic training.")
#
# NUM_FILTERS = 1
# graph_conv_filters =  A # you may try np.eye(3)
# graph_conv_filters = K.constant(graph_conv_filters)

# model = Sequential()
# model.add(GraphCNN(Y.shape[1], NUM_FILTERS, graph_conv_filters, input_shape=(X.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])
# model.summary()
#
# model.fit(X, Y, batch_size=A.shape[0], sample_weight=train_on_weight, epochs=200, shuffle=False, verbose=0)
# Y_pred = model.predict(X, batch_size=A.shape[0])
# print(np.argmax(Y_pred, axis=1))

# A = data[0]
# A_norm = utils.preprocess_adj_numpy(A, True)
# graph_conv_filters = A_norm
# graph_conv_filters = K.constant(graph_conv_filters)
# num_filters = 1
# X = np.ones(shape = (A.shape[0],A.shape[1],3))
# X[:,0] = np.sum(A, axis=1)
# y = np.sum(A, axis=1)
# train_mask = np.ones(shape=X.shape[0])
# for ind in [22,22,24,28,30,34]:
#     train_mask[ind]=0
#
# inputs1 = Input((3,))
#
# layer_conv1 = GraphCNN(2, num_filters, graph_conv_filters, activation='elu', kernel_regularizer=l2(5e-4))(inputs1)
# layer_conv2 = GraphCNN(2, num_filters, graph_conv_filters, activation='elu', kernel_regularizer=l2(5e-4))(layer_conv1)
# layer_conv3 = GraphCNN(2, num_filters, graph_conv_filters, activation='elu', kernel_regularizer=l2(5e-4))(layer_conv2)
# layer_conv4 = GraphCNN(2, num_filters, graph_conv_filters, activation='elu', kernel_regularizer=l2(5e-4))(layer_conv3)
# layer_conv5 = Maximum()([layer_conv1, layer_conv2, layer_conv3, layer_conv4])
# # layer_conv5 = Reshape((300,))(layer_conv5)
# # layer_conv5 = AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None)(layer_conv5)
# # layer_flatten = Flatten()(layer_conv5)
# # layer_fc1 = Dense(V - 1, activation=None)(layer_flatten)
# layer_fc1 = Activation('softmax')(layer_conv5)
# model = Model(inputs=inputs1, outputs=layer_fc1)
# model.summary()
#
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])
#
# nb_epochs = 100
# for epoch in range(nb_epochs):
#     model.fit(X, y, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
#
# Y_pred = model.predict(X, batch_size=A.shape[0])

#### learn betweenness rank with GCNN
# Generate data

Gd = GenerateData()
datadir = ".\data\processed\\adj_btw_pl_"
data, label, feature = Gd.generate_betdata_plmodel(n=5000, V=50, datadir=datadir, predictor="adjacency", genflag=0)
xtrain, ytrain, ftrain, xtest, ytest, ftest = Gd.splittwo_data(data, label, feature)

A, X, y = shuffle(xtrain, ftrain, ytrain)
Atest, Xtest, ytest = shuffle(xtest, ftest, ytest)

## filters
SYM_NORM = True
num_filters = 2
graph_conv_filters = preprocess_adj_tensor_with_identity(A, SYM_NORM)
graph_conv_filters_test = preprocess_adj_tensor_with_identity(Atest, SYM_NORM)

# model

X_input = Input(shape=(X.shape[1], X.shape[2]))
graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))

layer_gcnn1 = MultiGraphCNN(16, num_filters, activation='elu')([X_input, graph_conv_filters_input])
layer_gcnn1 = Dropout(0.2)(layer_gcnn1)
layer_gcnn2 = MultiGraphCNN(16, num_filters, activation='elu')([layer_gcnn1, graph_conv_filters_input])
layer_gcnn2 = Dropout(0.2)(layer_gcnn2)
layer_gcnn3 = MultiGraphCNN(16, num_filters, activation='elu')([layer_gcnn2, graph_conv_filters_input])
layer_gcnn3 = Dropout(0.2)(layer_gcnn3)
layer_gcnn4 = Maximum()([layer_gcnn1, layer_gcnn2, layer_gcnn3])
# layer_gcnn5 = Reshape((layer_gcnn4.shape[1]*layer_gcnn4.shape[2],))(layer_gcnn4)
layer_gcnn5 = Flatten()(layer_gcnn4)

# # layer_conv5 = AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None)(layer_conv5)
layer_dense1 = Dense(50, activation='linear')(layer_gcnn5)

nb_epochs = 500
batch_size = int(0.9*A.shape[0])
filepath ='.\models\\adj_btw_gcnn_pl.h5'
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
model = Model(inputs=[X_input, graph_conv_filters_input], outputs=layer_dense1)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit([X, graph_conv_filters], y, batch_size=batch_size, validation_split= 0.1, epochs=nb_epochs, callbacks=[mcp],
          shuffle=True, verbose=1)

# load best save model
model = load_model(filepath, custom_objects={'MultiGraphCNN': MultiGraphCNN})

# prediction

ypred = model.predict([Xtest, graph_conv_filters_test], steps=1)

### rank predictions
ypred = vs.rank_ypred(ypred, 1.8)

# visualize results
acc, precision, recall, f1score = vs.getacuracy(ytest, ypred)

