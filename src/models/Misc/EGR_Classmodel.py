import networkx as nx
import numpy as np
from src.data.make_dataset import GenEgrData
from src.features import build_features as bf
import os, sys
from keras.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning-master"))
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning-master/examples")) # Adding the submodule to the module search path
from utils import *
from keras_dgl.layers import GraphCNN, MultiGraphCNN
from src.models import cnn_tfmodel
ge = GenEgrData()

# generate data
datadir = ".\data\processed\\adj_egrclass_pl_"
predictor, label, feature = ge.gen_egr_plmodel(10000, 50, datadir, "adjacency", genflag=1)

# classigy labels into three class
label = bf.classifylabels(label)

# reshape data
predictor = np.reshape(predictor, (predictor.shape[0], predictor.shape[1], predictor.shape[2], 1))

# split data
xtrain, ytrain, ftrain, xtest, ytest, ftest = ge.split_data(predictor, label, feature)
xtrain, ytrain, ftrain, xval, yval, fval = ge.split_data(xtrain, ytrain, ftrain)

# call CNN regression model

model = cnn_tfmodel.cnnmodel2(50, 49)

# compile model
model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])

# train model
filepath ='.\models\\adj_egr_cnn_pl.h5'
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
model.fit(xtrain, ytrain, validation_data=[xval, yval], epochs=50, callbacks=[mcp], verbose=1)

# predict model
ypred = model.predict(xtest, steps=1)




