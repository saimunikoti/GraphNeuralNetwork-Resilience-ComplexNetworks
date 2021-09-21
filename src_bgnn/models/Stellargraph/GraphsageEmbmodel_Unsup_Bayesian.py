import pandas as pd
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UnsupervisedSampler
from tensorflow.keras import optimizers, losses, metrics, Model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import stellargraph as sg

## ================== define graph ==================

G = StellarGraph.from_networkx(gobs, node_features="feature")
print(G.info())

## =================== defined unsup sample =========

nodes = list(G.nodes())
number_of_walks = 1
length = 5

unsupervised_samples = UnsupervisedSampler(
    G, nodes=nodes, length=length, number_of_walks=number_of_walks)

## ====================== create a node pair generator =======================
batch_size = 50
epochs = 30
num_samples = [10, 5]

generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
train_gen = generator.flow(unsupervised_samples)

layer_sizes = [50, 50]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2")

# Build the model and expose input and output sockets of graphsage, for node pair inputs:
x_inp, x_out = graphsage.in_out_tensors()

#node classfiication layer
prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip")(x_out)

# model training
model = Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=optimizers.Adam(lr=1e-3),
    loss= losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy],
)

fileext = "\\plc_2000_egr_bayesian"

filepath ='.\models\\BayesianGraphsage' + fileext + '_unsupvisedemb.h5'
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(
    train_gen,
    epochs=epochs,
    callbacks=[mcp],
    verbose=1,
    use_multiprocessing=False,
    validation_data= train_gen,
    workers=4,
    shuffle=True)

sg.utils.plot_history(history)

## ================== Node embeddding ===================

x_inp_src = x_inp[0::2]
x_out_src = x_out[0]

embedding_model = Model(inputs=x_inp_src, outputs=x_out_src)

node_ids = targetdf.index
node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)

node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

# save embedding
tempdic = {}
tempdic['emb'] = node_embeddings
filename = config.datapath + "Bayesian"+ fileext+ "_unsupembeddings.mat"
io.savemat(filename, tempdic)

## ======================= visualize embedding =========================

targetdf['label'] = np.NaN
targetdf.loc[targetdf['metric']>=0.8,'label']= 3
targetdf.loc[(targetdf['metric']>=0.6) & (targetdf['metric']<0.8),'label']= 2
targetdf.loc[(targetdf['metric']>=0.3) & (targetdf['metric']<0.6),'label']= 1
targetdf.loc[(targetdf['metric']>=0) & (targetdf['metric']< 0.3),'label']= 0

X = node_embeddings
trans = TSNE(n_components=2)
emb_transformed = pd.DataFrame(trans.fit_transform(X), index=node_ids)
emb_transformed["label"] = targetdf['label']

alpha = 0.7

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    c=emb_transformed["label"].astype("category"),
    cmap="jet",
    alpha=alpha,
)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title(
    "{} visualization of GraphSAGE embeddings for cora dataset".format(transform.__name__)
)
plt.show()