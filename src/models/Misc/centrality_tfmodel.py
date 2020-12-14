
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from src.data import make_dataset as ut
from src.models.cnn_tfmodel import conv2d, maxpool, relu_activation, fclayer
from src.features import build_features as bf

# generate data
datadir = r"C:\Users\saimunikoti\Manifestation\centrality_learning\data\processed\\adj_btw_pl_"
data, label = ut.GenerateData().generate_betdata_plmodel(20000, datadir, "adjacency", genflag=1) # flag=0 for loading saved data
data = np.reshape(data, (data.shape[0],data.shape[1],data.shape[2],1))
xtrain, ytrain, xval, yval, xtest, ytest = ut.GenerateData().split_data(data, label)

# Placeholder variable for the input data
x = tf.placeholder(tf.float32, shape=[None, 22,22,1], name='X')
# Reshape it into [num_images, img_height, img_width, num_channels]
#x_image = tf.reshape(x, [-1, 22, 22, 1])

# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, 22], name='y_true')
#y_true_cls = tf.argmax(y_true, dimension=1)

# Convolutional Layer 1
layer_conv1, weights_conv1 = conv2d(input=x, num_input_channels=1, filter_size=3, num_filters=8, name ="conv1")
# Pooling Layer 1
layer_pool1 = maxpool(layer_conv1, name="pool1")
# RelU layer 1
layer_relu1 = relu_activation(layer_pool1, name="relu1")
# Convolutional Layer 2
layer_conv2, weights_conv2 = conv2d(input=layer_relu1, num_input_channels=8, filter_size=3, num_filters=16, name= "conv2")
# Pooling Layer 2
layer_pool2 = maxpool(layer_conv2, name="pool2")
# RelU layer 2
layer_relu2 = relu_activation(layer_pool2, name="relu2")
# Convolutional Layer 2
layer_conv3, weights_conv3 = conv2d(input=layer_relu2, num_input_channels=16, filter_size=3, num_filters=16, name= "conv3")
# Pooling Layer 2
layer_pool3 = maxpool(layer_conv3, name="pool3")
# RelU layer 2
layer_relu3 = relu_activation(layer_pool3, name="relu3")
# Flatten Layer
num_features = layer_relu3.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_relu3, [-1, num_features])
# Fully-Connected Layer 1
layer_fc1 = fclayer(layer_flat, num_inputs=num_features, num_outputs=22, name="fc1")

# cost function
with tf.name_scope("mse"):
    cost = tf.losses.mean_squared_error(y_true, layer_fc1)
    cost = tf.reduce_mean(cost)

# optimizer Use Adam Optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

# Accuracy
with tf.name_scope("accuracy"):
    lossvalue = tf.keras.losses.MSE(y_true, layer_fc1)
    accuracy = tf.reduce_mean(tf.cast(lossvalue, tf.float32))

# Initialize the FileWriter
writer = tf.summary.FileWriter("Training_FileWriter/")
writer1 = tf.summary.FileWriter("Validation_FileWriter/")

# Add the cost and accuracy to summary
tf.summary.scalar('loss', cost)
#tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

num_epochs = 100
batch_size = xtrain.shape[0]

filepath ='.\models\\tfmodel\\adj_btw_model_pl'

tf.add_to_collection('vars', layer_fc1)

# Session of tensorflow
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    #writer.add_graph(sess.graph)
    saver = tf.train.Saver()
    # Loop over number of epochs
    for epoch in range(num_epochs):

        start_time = time.time()
        train_accuracy = 0

    # for batch in range(0, int(xtrain.shape[0] / batch_size)):
        # Get a batch of images and labels
        #x_batch, y_true_batch = data.train.next_batch(batch_size)
        x_batch = xtrain
        y_true_batch = ytrain

        # Put the batch into a dict with the proper names for placeholder variables
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        sess.run(optimizer, feed_dict=feed_dict_train)

        # Calculate the accuracy on the batch of training data
        train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)

        # Generate summary with the current batch of data and write to file
        #summ = sess.run(merged_summary, feed_dict=feed_dict_train)
        #writer.add_summary(summ, epoch * int(len(data.train.labels) / batch_size) + batch)

        train_accuracy /= int(xtrain.shape[0]/ batch_size)

        # Generate summary and validate the model on the entire validation set
        val_accuracy = sess.run([accuracy], feed_dict={x: xval, y_true: yval})

        #writer1.add_summary(summ, epoch)

        end_time = time.time()

        print("Epoch " + str(epoch + 1) + " completed : Time usage " + str(int(end_time - start_time)) + " seconds")
        print("\t- Training Accuracy:\t{}".format(train_accuracy))
        print("\t- Validation Accuracy:\t{}".format(val_accuracy))

        saver.save(sess, filepath, global_step= 50)



