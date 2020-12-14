import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import models, layers
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Multiply, Lambda, Dropout, Maximum
import numpy as np
import keras.backend as K
from utils import *
from keras_dgl.layers import GraphCNN, MultiGraphCNN

def conv2d(self,input, num_input_channels, filter_size, num_filters, name):

    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        # TensorFlow operation for convolution
        layer = tf.keras.layers.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # Add the biases to the results of the convolution.
        layer += biases

        return layer, weights

#function for creatinf pooling layer
def maxpool(self,input, name, k=2):
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool(value=input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        return layer

# function for creating activation layer relu
def relu_activation(self,input, name):
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)

        return layer

def linear_activation(self,input, name):
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.linear(input)

        return layer

# function for creating FC layer
def fclayer(self,input, num_inputs, num_outputs, name):
    with tf.variable_scope(name) as scope:
        # Create new weights and biases.
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

        # Multiply the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        return layer

def cnnmodel1():

    model = models.Sequential()
    model.add(layers.Conv2D(8, kernel_size=3, activation='relu', input_shape=(22, 22,1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    # model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(22, activation='linear'))

    return model
# simple CNN regression model
def cnnmodel2(input_dim, out_dim):

    inputs1 = Input((input_dim, input_dim, 1))
    layer_conv1 = Conv2D(8, (3, 3), activation=None, padding='same', name='conv1')(inputs1)
    layer_conv1 = Activation('relu')(layer_conv1)
    layer_conv1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(layer_conv1)

    layer_conv2 = Conv2D(16, (3, 3), activation=None, padding='same', name='conv2')(layer_conv1)
    layer_conv2 = Activation('relu')(layer_conv2)
    layer_conv2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(layer_conv2)

    layer_conv3 = Conv2D(16, (2, 2), activation=None, padding='same', name='conv3')(layer_conv2)
    layer_conv3 = Activation('relu')(layer_conv3)
    layer_conv3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(layer_conv3)

    layer_flatten = Flatten()(layer_conv3)
    layer_flatten = Dropout((0.2))(layer_flatten)
    layer_fc1 = Dense(out_dim, activation=None)(layer_flatten)
    layer_fc1 = Activation('linear')(layer_fc1)

    model = Model(inputs=inputs1, outputs=layer_fc1)

    return model

def get_adjmask(inputdim, adjmatrices):
    inpadjtensor = []
    adjmatrices = np.reshape(adjmatrices,(adjmatrices.shape[0], adjmatrices.shape[1],adjmatrices.shape[2]))

    size = adjmatrices[0].shape[1]
    for adjmat in adjmatrices:
        adjtensor = np.zeros((size, size, inputdim))
        for countfilter in range(inputdim):
            adjtensor[:,:,countfilter] = adjmat

        inpadjtensor.append(adjtensor)

    inpadjtensor = np.array(inpadjtensor)
    inpadjtensor = inpadjtensor.astype('float32')
    inpadjtensor = tf.convert_to_tensor(inpadjtensor, dtype=tf.float32)

    return inpadjtensor

## mask output of each layer with Adjacency matrix
def cnnmodel_custom(adjmask, V):

    # def getadjmask(x):
    #     return x[0] * x[1][:, :, :, 0:8]

    inputs1 = Input((V, V, 1))
    mask_train = Input(shape=(V, V, 16), dtype='float32', name='mask')
    layer_conv1 = Conv2D(8, (2, 2), activation=None, strides=(1, 1), padding='same', name='conv1')(inputs1)
    layer_conv1 = Activation('relu')(layer_conv1)
    # layer_conv1 = Lambda(getadjmask)([layer_conv1, adjmask])

    layer_conv1 = Lambda(lambda x: x[0]*x[1][:, :, :, 0:8])([layer_conv1, mask_train])

    layer_conv2 = Conv2D(16, (3, 3), activation=None, strides=(1, 1), padding='same', name='conv2')(layer_conv1)
    layer_conv2 = Activation('relu')(layer_conv2)
    # layer_conv2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(layer_conv2)
    layer_conv2 = Lambda(lambda x: x[0]*x[1])([layer_conv2, mask_train])

    layer_conv3 = Conv2D(4, (2, 2), activation=None, strides=(1, 1), padding='same', name='conv3')(layer_conv2)
    layer_conv3 = Activation('relu')(layer_conv3)
    layer_conv3 = MaxPooling2D((3, 3), strides=(2, 2), name='pool3')(layer_conv3)
    # layer_conv3 = Lambda(lambda x: x[0] * x[1])([layer_conv3, mask_train])

    layer_flatten = Flatten()(layer_conv3)

    def sortnonzero(layer):
        return tf.math.top_k(layer, k=1824, sorted=True).values

    # layer_flatten = Lambda(sortnonzero)(layer_flatten)

    layer_fc1 = Dense(V, activation=None)(layer_flatten)
    layer_fc1 = Activation('linear')(layer_fc1)

    layer_fc1 = Dropout(0.2)(layer_fc1)

    model = Model(inputs=[inputs1, mask_train], outputs=layer_fc1)

    model.summary()
    return model

# loss functions for edge ranking
def edgerank_loss(y_true, y_pred):

    onetensor = tf.ones(shape=tf.shape(y_true))
    tempmatrix = K.dot(y_true, K.log(tf.transpose(y_pred))) + K.dot((onetensor - y_true),
                                                                    K.log(tf.transpose(onetensor - y_pred)))
    return (-1) * K.mean(tf.diag_part(tempmatrix))

def asymmetric_loss(alpha):
    def loss(y_true, y_pred):
        delta = y_pred - y_true
        return K.mean(K.square(delta) *
                      K.square(K.sign(delta) + alpha),
                      axis=-1)

    return loss

def noderankloss_v0(index, y_true, y_pred):
    # index = expandy(50)
    yt = y_true[:, index[:, 0]] - y_true[:, index[:, 1]]
    yp = y_pred[:, index[:, 0]] - y_pred[:, index[:, 1]]
    yp = np.log(yp)

    onetensor = tf.ones(shape=tf.shape(yt))
    tempmatrix = K.dot(yt, K.log(tf.transpose(yp))) + K.dot((onetensor - yt), K.log(tf.transpose(onetensor - yp)))

    return (-1) * K.mean(tf.diag_part(tempmatrix))

# checking loss functions for node ranking
def noderankloss(index):

    def loss(y_true, y_pred):
        # index = expandy(50)
        # yt = y_true[:,index[:, 0]] - y_true[:, index[:, 1]]
        yt = tf.math.sigmoid(tf.gather(y_true, index[:, 0], axis=1) - tf.gather(y_true, index[:, 1], axis=1))
        yp = tf.math.sigmoid(tf.gather(y_pred, index[:, 0], axis=1) - tf.gather(y_pred, index[:, 1], axis=1))
        # yt = y_true[:,index[:, 0]] - y_true[:, index[:, 1]]
        # yp = y_pred[:,index[:, 0]] - y_pred[:, index[:, 1]]

        onetensor = tf.ones(shape=tf.shape(yt))
        tempmatrix = (-1)*K.dot(yt, K.log(tf.transpose(yp))) - K.dot((onetensor - yt),
                                                                K.log(tf.transpose(onetensor - yp)))
        # tempmatrix = tf.Print(tempmatrix, [yt, yp], "...")
        return K.mean(tf.diag_part(tempmatrix))

    return loss

# modified form of node rank loss for avoiding Nan output(includes print option for loss function debugging)
def noderankloss_v2(index):

    def loss(y_true, y_pred):

        yt = tf.gather(y_true, index[:, 0], axis=1) - tf.gather(y_true, index[:, 1], axis=1)
        yp = tf.gather(y_pred, index[:, 0], axis=1) - tf.gather(y_pred, index[:, 1], axis=1)

        onetensor = tf.ones(shape=tf.shape(yt))

        zerotensor = tf.zeros(shape= (tf.shape(yt)[0], tf.shape(yt)[0]))

        yp_transpose = tf.transpose(yp)

        tempmatrix = tf.math.maximum(K.dot(onetensor, yp_transpose), zerotensor) - K.dot(yt, yp_transpose) - \
                     K.dot(onetensor, tf.math.log_sigmoid(tf.math.abs(yp_transpose)))

        # print intermediate tensor outputs but the print should be attached to node which is used further
        tempmatrix = tf.Print(tempmatrix, [tf.shape(onetensor), tf.shape(zerotensor) ], "...")

        return K.mean(tf.diag_part(tempmatrix))

    return loss

def egrmodel1(V):

    inputs1 = Input((V, V, 1))
    layer_conv1 = Conv2D(8, (3, 3), activation=None, padding='same', name='conv1')(inputs1)
    layer_conv1 = Activation('relu')(layer_conv1)
    layer_conv1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(layer_conv1)

    layer_conv2 = Conv2D(8, (3, 3), activation=None, padding='same', name='conv2')(layer_conv1)
    layer_conv2 = Activation('relu')(layer_conv2)
    layer_conv2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(layer_conv2)

    layer_conv3 = Conv2D(8, (2, 2), activation=None, padding='same', name='conv3')(layer_conv2)
    layer_conv3 = Activation('relu')(layer_conv3)
    layer_conv3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(layer_conv3)

    layer_flatten = Flatten()(layer_conv3)
    layer_fc1 = Dense(V-1, activation=None)(layer_flatten)
    layer_fc1 = Activation('sigmoid')(layer_fc1)

    model = Model(inputs=inputs1, outputs=layer_fc1)

    return model

def egrmodel2(A,X, graph_conv_filters,num_filters):

    X_input = Input(shape=(X.shape[1], X.shape[2]))
    graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))

    layer_gcnn1 = MultiGraphCNN(16, num_filters, activation='elu')([X_input, graph_conv_filters_input])
    layer_gcnn1 = Dropout(0.2)(layer_gcnn1)
    layer_gcnn2 = MultiGraphCNN(16, num_filters, activation='elu')([layer_gcnn1, graph_conv_filters_input])
    layer_gcnn2 = Dropout(0.2)(layer_gcnn2)
    layer_gcnn3 = MultiGraphCNN(16, num_filters, activation='elu')([layer_gcnn2, graph_conv_filters_input])
    layer_gcnn3 = Dropout(0.2)(layer_gcnn3)
    layer_gcnn4 = Maximum()([layer_gcnn1, layer_gcnn2, layer_gcnn3])
    # layer_gcnn5 = Reshape((layer_gcnn4.shape[1]*layer_gcnn4.shape[2],))(layer_gcnn4)
    layer_gcnn5 = Flatten()(layer_gcnn4)

    # # layer_conv5 = AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None)(layer_conv5)
    layer_dense1 = Dense(50, activation='sigmoid')(layer_gcnn5)

    model = Model(inputs=[X_input, graph_conv_filters_input], outputs=layer_dense1)
    model.summary()

    return model






