
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanPoolingAggregator, MaxPoolingAggregator, MeanAggregator, AttentionalAggregator
from tensorflow.keras import layers, optimizers, losses, metrics, Model, models
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn import model_selection
from src.data import config
import numpy as np
import tensorflow as tf
import stellargraph as sg
import scipy.io as io

## = ########################################### build graph ###############################################
#%% ############################################################################################################

G = StellarGraph.from_networkx(gobs, node_features="feature")
print(G.info())

train_subjects, test_subjects = model_selection.train_test_split(targetdf, train_size=0.9, test_size=None)

# temp_train_subjects = np.reshape(np.array(train_subjects), (train_subjects.shape[0],1))
# temp_test_subjects = np.reshape(np.array(test_subjects), (test_subjects.shape[0],1))
# train_targets = target_encoding.fit_transform(temp_train_subjects).toarray()
# test_targets = target_encoding.transform(temp_test_subjects).toarray()

train_targets = np.array(train_subjects)
test_targets = np.array(test_subjects)

## #################################### Graphsage Model building ###########################################
#%% ############################################################################################################

batch_size = 40
# number of nodes to consider for each hop
num_samples = [15, 10, 5]
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)  # train_subjects.index for selecting training nodes
test_gen = generator.flow(test_subjects.index, test_targets)

graphsage_model = GraphSAGE(layer_sizes=[64, 32, 16], generator=generator, activations=["relu","relu","linear"],
                            bias=True, dropout=0.0)

x_inp, x_out = graphsage_model.in_out_tensors()
x_outdense = layers.Dense(units=10, activation="relu")(x_out)
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_outdense)

model = Model(inputs=x_inp, outputs=prediction)
model.summary()
##
model.compile( optimizer=optimizers.RMSprop(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])

fileext = "\\plc_5000_egr_bayesian"

filepath ='.\models\\BayesianGraphsage' + fileext + '_supvisedemb.h5'

mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
weights = {0:100, 1:10, 2:1}
history = model.fit(train_gen, class_weight=weights, epochs=60, validation_data = test_gen, callbacks=[mcp], verbose=2, shuffle=False)

## ================== Node embeddding ===================
sg.utils.plot_history(history)

embedding_model = Model(inputs=x_inp, outputs=x_out)
embedding_model.summary()

node_ids = targetdf.index
node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)

node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

# save embedding
tempdic = {}
tempdic['emb'] = node_embeddings
filename = config.datapath + "Bayesian"+ fileext+ "_supembeddings.mat"
io.savemat(filename, tempdic)

