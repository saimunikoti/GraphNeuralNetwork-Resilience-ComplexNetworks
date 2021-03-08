
from stellargraph import StellarGraph
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from tensorflow.keras.layers import Dense, Dropout, Input
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanPoolingAggregator, MaxPoolingAggregator, MeanAggregator, AttentionalAggregator
import seaborn as sns
from tensorflow.keras import layers, optimizers, losses, metrics, Model, models
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn import preprocessing, feature_extraction, model_selection
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
## = ########################################### build graph ###############################################
#%% ############################################################################################################

G = StellarGraph.from_networkx(g, node_features="feature")
print(G.info())

train_subjects, test_subjects = model_selection.train_test_split(targetdf, train_size=0.8, test_size=None)

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

# aggregatortype = MaxPoolingAggregator(),
# layer_sizes (list): Hidden feature dimensions for each layer. activations (list): Activations applied to each layer's output;

def get_dropout(input_tensor, p=0.1, mc=False):
   if mc:
      return Dropout(p)(input_tensor, training=True)
   else:
      return Dropout(p)(input_tensor)

graphsage_model = GraphSAGE(layer_sizes=[64, 32, 16], generator=generator, activations=["relu","relu","linear"],
                            bias=True, aggregator = MeanAggregator, dropout=0.0)

x_inp, x_out = graphsage_model.in_out_tensors()
x_out = layers.Dense(units=10, activation="relu")(x_out)
# x_out = get_dropout(x_out, p=0.1, mc='mc')
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=prediction)
model.summary()

##
model.compile( optimizer=optimizers.Adam(), loss = noderankloss(), metrics=["acc"])
model.compile( optimizer=optimizers.Adam(), loss="mean_squared_error", metrics=["acc"])
model.compile( optimizer=optimizers.RMSprop(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])

filepath ='.\models\\BayesianGraphsage' + fileext + '_classf_meanagg.h5'

mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(train_gen, epochs = 50, validation_data=test_gen, callbacks=[mcp], verbose=2, shuffle=False)

from tensorflow.keras.models import load_model
model = load_model(filepath, custom_objects={"MeanAggregator": MeanAggregator})

all_nodes = targetdf.index
all_mapper = generator.flow(all_nodes)
y_pred = model.predict(all_mapper)

test_targets = np.array(targetdf)
y_test = np.argmax(test_targets, axis=1)
y_pred = np.argmax(y_pred, axis=1)

cnfmatrix = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred, average="weighted")
recall_score(y_test, y_pred, average="weighted")
class0accuracy = cnfmatrix[0][0]/np.sum(cnfmatrix[0,:])
mc_predictions = []

import tqdm
for i in tqdm.tqdm(range(100)):
    y_p = model.predict(all_mapper)
    mc_predictions.append(y_p)

def get_ranksfrommetic( mc_predictions):
   listnoderanks=[]
   acclist = []
   for count in range(mc_predictions.shape[0]):
        temp_pred = np.argmax(mc_predictions[count], axis=1)
        acclist.append(accuracy_score(y_test, temp_pred))
        listnoderanks.append(temp_pred)

   return np.array(listnoderanks), acclist

mc_predictions = np.array(mc_predictions)

mc_classpredictions, mc_accuracy = get_ranksfrommetic(mc_predictions)

mc_ensemble_predictions = np.mean(mc_classpredictions, axis=0)
accuracy_score(y_test, mc_ensemble_predictions)

## ================= plot ========================

fig1, ax1 = plt.subplots(nrows=2, ncols=2, sharex=False,sharey=False,figsize=(8,6))
nodepred = mc_predictions[:,2,:]
sns.kdeplot(ax=ax1[0,0], x=nodepred[:, 0], shade=True)
ax1[0,0].set_title("Class 0")
nodepred = mc_predictions[:,126,:]
sns.kdeplot(ax=ax1[0,1], x=nodepred[:, 0], shade=True)
ax1[0,1].set_title("Class 0")
nodepred = mc_predictions[:,127,:]
sns.kdeplot(ax=ax1[1,0], x=nodepred[:, 1], shade=True)
ax1[1,0].set_title("Class 1")
nodepred = mc_predictions[:,125,:]
sns.kdeplot(ax=ax1[1,1], x=nodepred[:, 2], shade=True)
ax1[1,1].set_title("Class 2")

fig1.suptitle('Variation of class probability for nodes from different class', fontsize=18)
fig1.text(0.5, 0.005, 'Class probability', ha='center', fontsize=16)
# fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')

##

