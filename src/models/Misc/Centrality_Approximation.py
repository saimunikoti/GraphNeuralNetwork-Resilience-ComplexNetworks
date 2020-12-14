import time
import scipy.sparse
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
import tensorflow as tf
import os
os.chdir(r"C:\Users\saimunikoti\Manifestation\centrality_learning\models\GCN_Regression")
import layers.graph as lg
import config.sparse as us


#g = nx.read_graphml('R/karate.graphml')

### generate power law graph
g = nx.generators.random_graphs.powerlaw_cluster_graph(12,1,0.4)
g1 = nx.generators.random_graphs.powerlaw_cluster_graph(12,1,0.4, seed =120)


### load graph data

# g = nx.DiGraph()
# g.add_weighted_edges_from([(6,0,0.2),(7,0,0.8),(4,1,1),(5,3,0.9),(8,3,0.1),(11,3,0.2),(12,3,0.8), (9,4,1),(0,5,1),(2,6,1),
#                             (4,7,1),(1,8,1),(1,9,1),(4,10,1),(2,11,1),(4,12,1),(11,13,0.2),(12,13,0.8),(13,14,1)])


## degree histogram
Degree = nx.degree(g)
degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
plt.figure()
plt.bar(deg, cnt, width=0.80, color='cornflowerblue')

adjg = nx.adj_matrix(g)
lapg = nx.laplacian_matrix(g)
degrg = lapg+adjg
adjg1 = nx.adj_matrix(g1)
lapg1 =nx.laplacian_matrix(g1)
degrg1= lapg1 + adjg1
# Get important parameters of adjacency matrix
n_nodes = adjg.shape[0]

# Some preprocessing
def get_adjnorm(adj):

    # adj_tilde = np.absolute(adj)
    adj_tilde = adj + np.identity(n=adj.shape[0])
    d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
    # d_tilde_diag = np.squeeze(np.sum(adj_tilde, axis=1))
    d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2)
    d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
    adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
    adj_norm_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))

    return adj_norm_tuple

adj_norm_tuple_train = get_adjnorm(adjg)
adj_norm_tuple_val = get_adjnorm(adjg1)

# Features are just the identity matrix
# feat_x = degrg
# feat_x_val = degrg1
feat_x = np.identity(n=adjg.shape[0])
feat_x_val = np.identity(n=adjg.shape[0])

feat_x_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(feat_x))
feat_x_tuple_val = us.sparse_to_tuple(scipy.sparse.coo_matrix(feat_x_val))

# get betweenness score of the graph as target variable
rankstrain = dict(nx.betweenness_centrality(g))
rankstest = dict(nx.betweenness_centrality(g1))

def get_label(rankstrain):
    for keys in rankstrain:
        if rankstrain[keys]>0.2:
            rankstrain[keys]=2
        elif rankstrain[keys]>0 and rankstrain[keys]<=0.2:
            rankstrain[keys]=1
        else:
            rankstrain[keys] = 0

    return rankstrain

rankstrain = get_label(rankstrain)
rankstest = get_label(rankstest)

# assign betweenness score to each node
for node in g.nodes:
    g.nodes[node]['centrality'] = rankstrain[node]
    g1.nodes[node]['centrality'] = rankstest[node]

# Semi-supervised
memberships = [m for m in nx.get_node_attributes(g, 'centrality').values()]

nb_classes = len(set(memberships))
targets = np.array([memberships], dtype=np.int32).reshape(-1)
one_hot_targets = np.eye(nb_classes)[targets]
y_train = one_hot_targets.copy()

memberships = [m for m in nx.get_node_attributes(g1, 'centrality').values()]

targets = np.array([memberships], dtype=np.int32).reshape(-1)
y_val = np.eye(nb_classes)[targets]

# Pick one at random from each class
# labels_to_keep = [np.random.choice(
#     np.nonzero(one_hot_targets[:, c])[0]) for c in range(nb_classes)]

# labels_to_keep = [0,3,5,8,10,13,16,20,25,28,29,30,32]
# labels_to_keep = list(np.arange(0,34))

# y_train = np.zeros(shape=one_hot_targets.shape,
#                    dtype=np.float32)


# y_val = one_hot_targets.copy()

# train_mask = np.zeros(shape=(n_nodes,), dtype=np.bool)
# val_mask = np.ones(shape=(n_nodes,), dtype=np.bool)

# for l in labels_to_keep:
#     y_train[l, :] = one_hot_targets[l, :]
#     y_val[l, :] = np.zeros(shape=(nb_classes,))
#     train_mask[l] = True
#     val_mask[l] = False

# TensorFlow placeholders
ph = {
    'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_mat"),
    'x': tf.sparse_placeholder(tf.float32, name="x"),
    'labels': tf.placeholder(tf.float32, shape=(n_nodes, nb_classes))}
    # 'mask': tf.placeholder(tf.int32)}

l_sizes = [6, 5, 3, nb_classes]

o_fc1 = lg.GraphConvLayer(
    input_dim=feat_x.shape[-1],
    output_dim=l_sizes[0],
    name='fc1',
    activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=ph['x'], sparse=True)

o_fc2 = lg.GraphConvLayer(
    input_dim=l_sizes[0],
    output_dim=l_sizes[1],
    name='fc2',
    activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc1)

o_fc2 = tf.nn.dropout(o_fc2, keep_prob = 0.8)

o_fc3 = lg.GraphConvLayer(
    input_dim=l_sizes[1],
    output_dim=l_sizes[2],
    name='fc3',
    activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc2)

# o_fc4 = lg.GraphConvLayer(
#     input_dim=l_sizes[2],
#     output_dim=l_sizes[3],
#     name='fc4',
#     activation=tf.identity)(adj_norm=ph['adj_norm'], x=o_fc3)

## mlp
o_fc4 = tf.layers.dense(o_fc3, nb_classes,activation=tf.nn.relu)

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def accuracy(preds, labels):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    # mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

with tf.name_scope('optimizer'):
    loss = softmax_cross_entropy(
        preds=o_fc4, labels=ph['labels'])

    accuracy = accuracy(
        preds=o_fc4, labels=ph['labels'])

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1e-2)

    opt_op = optimizer.minimize(loss)

feed_dict_train = {ph['adj_norm']: adj_norm_tuple_train,
                   ph['x']: feat_x_tuple,
                   ph['labels']: y_train}
                   # ,ph['mask']: train_mask}

feed_dict_val = {ph['adj_norm']: adj_norm_tuple_val,
                 ph['x']: feat_x_tuple_val,
                 ph['labels']: y_val}
                 # ,ph['mask']: val_mask}

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 60001
save_every = 2000

t = time.time()
outputs = {}
# Train model
for epoch in range(epochs):
    # Construct feed dictionary
    # Training step
    _, train_loss, train_acc = sess.run(
        (opt_op, loss, accuracy), feed_dict = feed_dict_train)

    if epoch % save_every == 0:
        # Validation
        val_loss, val_acc, op_4 = sess.run((loss, accuracy, o_fc4), feed_dict=feed_dict_val)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_acc),
              "val_loss=", "{:.5f}".format(val_loss),
              "val_acc=", "{:.5f}".format(val_acc),
              "time=", "{:.5f}".format(time.time() - t))

        feed_dict_output = {ph['adj_norm']: adj_norm_tuple_val,
                            ph['x']: feat_x_tuple_val}

        output = sess.run(o_fc3, feed_dict=feed_dict_output)
        outputs[epoch] = output

# model testing
# val_loss, val_acc = sess.run((o_fc3), feed_dict=feed_dict_val)
print("model performance in acc", val_acc)
# Save Model
saver = tf.train.Saver()
saver.save(sess, './Models/model.ckpt')
sess.close()

### load saved model

# with tf.Session() as sess:
#     new_saver = tf.train.import_meta_graph('./Models/model.ckpt.meta')
#     new_saver.restore(sess, tf.train.latest_checkpoint('./Models/'))
#     sess.run((accuracy), feed_dict=feed_dict_val)
#   print("validation perf", val_loss, val_acc)





