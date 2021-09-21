import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
import networkx as nx
import pandas as pd
import os
import stellargraph as sg
from stellargraph.core.graph import StellarGraph
from stellargraph.mapper.sampled_node_generators_bayesian import GraphSAGENodeGenerator
from stellargraph.layer.graphsage_bayesian import GraphSAGE, MaxPoolingAggregator, MeanAggregator, MeanAggregatorvariance

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from src_bgnn.data import config as cnf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import random
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

## ======================build graph ###############################################

dataset = datasets.Cora()
display(HTML(dataset.description))
G, node_subjects = dataset.load()

print(G.info())

## generate probability of edges
g = G.to_networkx() # multi
g = nx.Graph(g) # simple graph

for (cn1, cn2) in g.edges:
    g[cn1][cn2]['weight'] = np.round(random.uniform(0.5, 1), 3)

G = StellarGraph.from_networkx(gobs, node_features="feature")
print(G.info())

train_subjects, test_subjects = model_selection.train_test_split(
    targetdf, train_size=0.90, test_size=None )

train_targets = np.array(train_subjects)
test_targets = np.array(test_subjects)
#
# train_subjects = train_subjects[0:800]
# test_subjects = test_subjects[0:1896]

target_encoding = preprocessing.LabelBinarizer()
train_targets = target_encoding.fit_transform(train_subjects)
test_targets = target_encoding.transform(test_subjects)

##
batch_size = 40
# number of nodes to consider for each hop
num_samples = [15, 10, 5]
#
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)  # train_subjects.index for selecting training nodes
test_gen = generator.flow(test_subjects.index, test_targets)

def get_dropout(input_tensor, p=0.1, mc=False):
   if mc:
      return Dropout(p)(input_tensor, training=True)
   else:
      return Dropout(p)(input_tensor)

## model building

# graphsage_model = GraphSAGE(layer_sizes=[32, 32, 16], generator=generator, bias=True, dropout=0.5)
graphsage_model = GraphSAGE(layer_sizes=[64, 32, 16], generator=generator,activations=["relu","relu","linear"],
                            bias=True, aggregator= MeanAggregator, dropout=0.1)

x_inp, x_out = graphsage_model.in_out_tensors()
x_out = layers.Dense(units=10, activation="relu")(x_out)
x_out = layers.Dense(units=10, activation="relu")(x_out)
# x_out = get_dropout(x_out, p=0.1, mc=True)
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=prediction)
model.summary()

## %% ##################################### Model training #######################################################

model.compile( optimizer=optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])

fileext = "plc_4000_egr_bgnn.h5"
filepath = cnf.modelpath + fileext
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
weights = {0:100, 1:10, 2:1}
history = model.fit(train_gen, epochs=30, validation_data = test_gen, callbacks=[mcp], verbose=2, shuffle=False)
sg.utils.plot_history(history)

from tensorflow.keras.models import load_model

model = load_model(filepath, custom_objects={"MeanAggregator": MeanAggregator})

## testing evaluation

all_nodes = targetdf.index
all_mapper = generator.flow(all_nodes)

y_pred = model.predict(all_mapper)

test_targets = np.array(targetdf)
y_test = np.argmax(test_targets, axis=1)
y_pred = np.argmax(y_pred, axis=1)

test_metrics = model.evaluate(test_gen)

print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

cnfmatrix = confusion_matrix(y_test, y_pred)
print(cnfmatrix)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred, average="weighted")
recall_score(y_test, y_pred, average="weighted")
class0accuracy = cnfmatrix[0][0]/np.sum(cnfmatrix[0,:])
print(class0accuracy)

# save report as df
print(classification_report(y_test, y_pred))
report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose()
filepath ='.\models\\BayesianGraphsage' + fileext + '_Gobsnoise_classreport.csv'
df.to_csv(filepath)

