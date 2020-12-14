import networkx as nx
import numpy as np
import tensorflow as tf
import pandas as pd
from src.data import make_dataset as ut
from src.visualization import visualize as vs
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Multiply
from keras.callbacks import  ModelCheckpoint
from src.features import build_features as bf
from src.models.cnn_tfmodel import Tfkeraslegos
import warnings
warnings.filterwarnings("ignore")

# generate data
datadir = ".\data\processed\\adj_betweenness_pl_"
data, label , feature = ut.GenerateData().generate_betweenness_plmodel(5000, 50, datadir, "adjacency", genflag=0)
data = np.reshape(data, (data.shape[0],data.shape[1],data.shape[2],1))
xtrain, ytrain, xval, yval, xtest, ytest = ut.GenerateData().split_data(data, label)

# get adjacency mask
tfkeras = Tfkeraslegos()
adjmask_train = tfkeras.get_adjmask(16,xtrain)
adjmask_test = tfkeras.get_adjmask(16,xtest)
adjmask_val = tfkeras.get_adjmask(16,xval)

# tf keras Model building
model = tfkeras.cnnmodel_custom(adjmask_train, 10)
# model = tfkeras.cnnmodel2()

model.compile(optimizer='adagrad', loss='mean_squared_error',
              metrics=['accuracy'])

filepath ='.\models\\adj_btw_cstmodel2_ab.h5'
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
model.fit(x=[xtrain, adjmask_train], y=ytrain, epochs=150,steps_per_epoch=1,\
    validation_data = ([xval, adjmask_val],yval), validation_steps=1, callbacks=[mcp])

# prediction
# model = model.load_model(filepath)
ypred= model.predict([xtest, adjmask_test], steps=1)
#
ypred = bf.get_tranformpred(ypred)
#
## performance
acc, precision, recall, f1score, cm = vs.getacuracy(ytest, ypred)
accuracy = np.mean(acc)

precision = np.mean(precision)
recall = np.mean(recall)

Resultsdf = pd.DataFrame(data= acc, columns="Accuracy")
Resultsdf['precision'] = precision
Resultsdf['recall'] = recall
Resultsdf['mean acc'] = np.mean(acc)
Resultsdf['mean precision'] = np.mean(precision)
Resultsdf['mean recall'] = np.mean(recall)

# # visualize plots
# vs.checkgraphvariation(xtrain, xtest)

# count same data in training and testing
# countsame = bf.inputvar(xtrain, xtest)

