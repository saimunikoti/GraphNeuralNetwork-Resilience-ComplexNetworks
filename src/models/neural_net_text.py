import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
import pickle

# declare parameters
k = 4
n = 1*k

# load in training data
xs = np.genfromtxt(r"C:\Users\saimunikoti\Manifestation\centrality_learning\src\models\d4_20dB_model0ones_fixedH_labels.csv", delimiter = ',')
ys = np.genfromtxt(r"C:\Users\saimunikoti\Manifestation\centrality_learning\src\models\d4_20dB_model0ones_fixedH_data.csv", delimiter = ',')

# xs = xs.T
# ys = ys.T

# split into training and test sets

## -==== convert target vector inot probbilities ===========
xs[xs==-1]=0

obs_train, obs_test, lab_train, lab_test = \
    train_test_split(ys,xs, test_size=0.8)

## define the neural net model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(100, activation = 'relu'),
#     tf.keras.layers.Dense(100, activation = 'relu'),
#     tf.keras.layers.Dense(100, activation = 'relu'),
#     tf.keras.layers.Dense(100, activation = 'relu'),
#     tf.keras.layers.Dense(100, activation = 'relu'),
#     tf.keras.layers.Dense(100, activation = 'relu'),
#     tf.keras.layers.Dense(100, activation = 'relu'),
#     tf.keras.layers.Dense(4, activation = 'tanh')
#     ])
##
# example of a model defined with the functional api

# define the layers
xin = Input(shape=(4,))
xout1 = Dense(100, activation = 'relu')(xin)
xout2 = Dense(100, activation = 'relu')(xout1)
xout3 = Dense(4, activation = 'sigmoid')(xout2)
# define the model
model = Model(inputs=xin, outputs=xout3)

##
# compile the model, defininig the optimizer and loss function
## customloss function for learning prob

#@tf.function
def binaryloss(y_true, y_pred):

    tf.print("y_true", y_true[0,:])
    tf.print("y_pred", y_pred[0,:])

    onetensor = tf.ones(shape=tf.shape(y_true))

    temploss = (-1)*tf.reduce_sum(tf.math.multiply(y_true, tf.math.log(y_pred))) - tf.reduce_sum(tf.math.multiply((onetensor - y_true), \
                                                                tf.math.log(onetensor - y_pred)))
    return temploss

##
model.compile(optimizer='adam', loss=binaryloss)

# fit the model to the training data and evaluate with the test data
tic     = time.perf_counter()
history = model.fit(obs_train, lab_train, epochs = 1, batch_size=32, verbose=1)
toc     = time.perf_counter()
model.evaluate(obs_test, lab_test)
print('Total training time: ' + str(toc - tic) + ' seconds.')


# save off the model and model history
# model.save('model0ones_15dB_fixedH')
# with open('F:/Comm Studies/PermutationSimulations/UnlabeledSensing/TrainingHistories/trainHistoryDict_model0ones_15dB_fixedH', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)

# -----------------------------------------------------------------------------
# ---------- MISCELLANEOUS TESTING CODE - IGNORE BELOW THIS POINT -------------
# -----------------------------------------------------------------------------

#---------------------------US test-------------------------------------------

#model = tf.keras.models.load_model('model0_100dB_fixedH')

# # test out model to get mean squared error
# tic = time.perf_counter()
# err = 0;
# for i in range(500):
#     x = np.random.rand(4,1)
#     A = np.genfromtxt('meas_mat.csv', delimiter = ',')
#     y = np.matmul(A,x)
#     y = y[np.random.permutation(list(range(0,8))),0].reshape((1,n))
#     xEst = model.predict(y)
#     err += (1/k) * np.sum( np.square( x.T - xEst ) )
# toc = time.perf_counter()  
# print('Time to estimate single sample: ' + str(toc - tic) + ' seconds.')
# print('Average MSE = ' + str(err/500))

# obs = y + np.random.normal(size = (8,1))
# obs = y.reshape((1,n))
# # #obs = np.random.rand(1,n)

# # tic = time.perf_counter()
# # prediction = model.predict(obs)
# # toc = time.perf_counter()

# #randomly permute obs and run prediction
# order = list(range(0,8))
# order_ = np.random.permutation(order)
# obs2 = obs[0,order_].reshape((1,n))
# pred2 = model.predict(obs2)

# print(x.T)
# print(pred2)

# print('Time to estimate single sample: ' + str(toc - tic) + ' seconds.')
# #prediction.T

# #-----------------------HF test-----------------------------------------------
# H = np.genfromtxt('Meas_Mat/d4_15dB_model0ones_fixedH_Hmat.csv', delimiter=',')

# SNR     = 15
# Es      = (2/3) * 10**( SNR / 10 )
# x       = np.array([[1], [-1], [-1], [-1]])
# x_amp   = x * (Es**0.5)
# y       = np.matmul(H,x_amp) + np.random.normal(size=(4,1))
# y       = y.reshape((1,4))

# order   = list(range(0,4))
# order_  = np.random.permutation(order)
# y       = y[0,order_].reshape((1,4))

# pred    = model.predict(y)

# print(x.T)
# print(pred)

# #-----------------------PERM EST TEST---------------------------------------
# A       = np.genfromtxt('Meas_Mat/stochEM_exp_Pi_est_meas_mat', delimiter=',')

# test    = obs_test[194,:].reshape((1,20))
# label   = lab_test[194,:]

# pred    = model.predict(test)

# print(pred)
# print(label)











