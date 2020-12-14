
import numpy as np
import tensorflow as tf
from src.data import make_dataset as ut
from src.features import build_features as bf

# generate data
datadir = r"C:\Users\saimunikoti\Manifestation\centrality_learning\data\processed\\adj_btw_pl_"
data, label = ut.GenerateData().generate_betdata_plmodel(20000, datadir, "adjacency", genflag=1)  # flag=0 for loading saved data
data = np.reshape(data, (data.shape[0],data.shape[1],data.shape[2],1))
xtrain, ytrain, xval, yval, xtest, ytest = ut.GenerateData().split_data(data, label)

# predict test sample
sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(".\models\\tfmodel\\adj_btw_model_pl-50.meta")
saver.restore(sess,tf.train.latest_checkpoint(".\models./"))
all_vars = tf.get_collection('vars')
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
y_true = graph.get_tensor_by_name("y_true:0")
feed_dict ={X:xtest,y_true: ytest}
# #Now, access the op that you want to run.
# op_to_restore = graph.get_tensor_by_name("relu3:0")
ypred = (sess.run(all_vars[0], feed_dict))
ypred = bf.get_tranformpred(ypred)

a = tf.constant([[[1,1],[3,6]],[[7,8],[3,3]]])
b= tf.where(tf.equal(a,0))

with tf.Session() as sess:
    output = sess.run(b)
    print(output)


