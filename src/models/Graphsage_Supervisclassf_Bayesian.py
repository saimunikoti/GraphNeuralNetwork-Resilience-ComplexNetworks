
from stellargraph import StellarGraph
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from tensorflow.keras.layers import Dense, Dropout, Input
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanPoolingAggregator, MaxPoolingAggregator, MeanAggregator, AttentionalAggregator
import seaborn as sns

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from tensorflow.keras import layers, optimizers, losses, metrics, Model, models
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn import preprocessing, feature_extraction, model_selection
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

## = ########################################### build graph ###############################################
#%% ############################################################################################################

G = StellarGraph.from_networkx(gobsnoise, node_features="feature")
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
                            bias=True, aggregator = MaxPoolingAggregator, dropout=0.1)

x_inp, x_out = graphsage_model.in_out_tensors()
x_out = layers.Dense(units=10, activation="relu")(x_out)
x_out = layers.Dense(units=10, activation="relu")(x_out)
x_out = get_dropout(x_out, p=0.1, mc='mc')
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=prediction)
model.summary()

##
# model.compile( optimizer=optimizers.Adam(), loss = noderankloss(), metrics=["acc"])
# model.compile( optimizer=optimizers.Adam(), loss="mean_squared_error", metrics=["acc"])
model.compile( optimizer=optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])

fileext = "\\plc_5000_egr_estsupbayesian_mc"

filepath ='.\models\\BayesianGraphsage' + fileext + '_classf_classweight.h5'

mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
weights = {0:100, 1:10, 2:1}
history = model.fit(train_gen, class_weight=weights, epochs = 50, validation_data=test_gen, callbacks=[mcp], verbose=2, shuffle=False)

from tensorflow.keras.models import load_model
model_est = load_model(filepath, custom_objects={"MeanAggregator": MeanAggregator})

fileext = "\\plc_5000_gobs_bayesian"

filepath ='.\models\\BayesianGraphsage' + fileext + '_classf_classweight.h5'
model_obs = load_model(filepath, custom_objects={"MeanAggregator": MeanAggregator})

all_nodes = targetdf.index
all_mapper = generator.flow(all_nodes)
start = time.time()
y_pred = model_est.predict(all_mapper)
end = time.time()
print(end - start)

y_pred = model_obs.predict(all_mapper)

test_targets = np.array(targetdf)
y_test = np.argmax(test_targets, axis=1)
y_pred = np.argmax(y_pred, axis=1)

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

## monte carlo prediction with dropout masks

mc_predictions = []
all_nodes = targetdf.index
all_mapper = generator.flow(all_nodes)

import tqdm
for i in tqdm.tqdm(range(100)):
    y_p = model_est.predict(all_mapper)
    mc_predictions.append(y_p)

def get_ranksfrommetic( mc_predictions):
   listnoderanks=[]
   acclist = []
   class1accuracy = []
   for count in range(mc_predictions.shape[0]):
        temp_pred = np.argmax(mc_predictions[count], axis=1)
        acclist.append(accuracy_score(y_test, temp_pred))
        listnoderanks.append(temp_pred)
        cnfmatrix = confusion_matrix(y_test, temp_pred)
        class1accuracy.append(cnfmatrix[0][0]/np.sum(cnfmatrix[0,:]))

   return np.array(listnoderanks), acclist, class1accuracy

mc_predictions = np.array(mc_predictions)

mc_classpredictions, mc_accuracy, class1accuracy = get_ranksfrommetic(mc_predictions)

Meanaccuracy = np.mean(mc_accuracy)
Varaccuracy = np.var(mc_accuracy)

mc_ensemble_predictions = np.mean(mc_predictions, axis=0)
y_pred_ensemble = np.argmax(mc_ensemble_predictions, axis=1)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

cnfmatrix = confusion_matrix(y_test, y_pred_ensemble)
print(cnfmatrix)
accuracy_score(y_test, y_pred_ensemble)
precision_score(y_test, y_pred_ensemble, average="weighted")
recall_score(y_test, y_pred_ensemble, average="weighted")
class1accuracy = cnfmatrix[0][0]/np.sum(cnfmatrix[0,:])
print(class1accuracy)
# save report as df
print(classification_report(y_test, y_pred_ensemble))

## class 1 confidence interval
class1_loc = np.where(y_test==0)[0]
mc_predicions_class1 = mc_predictions[:,class1_loc,:]
mc_predicions_class1 = np.max(mc_predicions_class1, axis=2)

## plot class1 samples prob
fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharex=False,sharey=False,figsize=(8,6))

sns.kdeplot(ax=ax1[0], x=mc_predicions_class1[:, 20], shade=True)
ax1[0].set_title("Class-1 Test sample 20", fontsize=18)
ax1[0].set_ylabel("")
ax1[0].tick_params(labelsize=18)
sns.kdeplot(ax=ax1[1], x=mc_predicions_class1[:, 30], shade=True)
ax1[1].set_title("Class-1 Test sample 30", fontsize=18)
ax1[1].set_ylabel("")
ax1[1].tick_params(labelsize=18)

# fig1.suptitle('Variation of class probability for class-1 (high criticality) nodes , fontsize=18)
fig1.text(0.5, 0.04, 'Class probability', ha='center', fontsize=21)
fig1.text(0.09, 0.5, 'Density', va='center', rotation='vertical', fontsize=21)
ax1[0].grid(True)
ax1[1].grid(True)

## ================= plot  all classes ========================

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

