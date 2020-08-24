import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import scipy.stats as stats
from scipy.stats import rankdata
from keras.callbacks import Callback
##### get performance metrics
def getacuracy(y_true, y_pred):
    cm = np.array([confusion_matrix(y_true[ind,:], y_pred[ind,:]) for ind in range(y_true.shape[0])])
    ac = np.array([accuracy_score(y_true[ind,:], y_pred[ind,:]) for ind in range(y_true.shape[0])])
    pr = np.array([precision_score(y_true[ind,:], y_pred[ind,:],average='weighted') for ind in range(y_true.shape[0])])
    rc = np.array([recall_score(y_true[ind,:], y_pred[ind,:],average='weighted') for ind in range(y_true.shape[0])])
    f1 = np.array([f1_score(y_true[ind,:], y_pred[ind,:],average='weighted') for ind in range(y_true.shape[0])])
    # pr = np.mean(pr)
    # rc = np.mean(pr)
    # f1 = np.mean(pr)
    # ac = np.mean(ac)
    
    return ac, pr,rc,f1

### check the variation of graph in training and test data
def checkgraphvariation(xtrain, xtest):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes.flatten()

    for i in range(4):
        train = np.reshape(xtrain[i+12],(xtest.shape[1],xtest.shape[1]))
        test = np.reshape(xtest[i+40],(xtest.shape[1],xtest.shape[1]))
        if i<2:
            Gtrain = nx.from_numpy_matrix(train)
            pos= nx.circular_layout(Gtrain)       
            nx.draw_networkx(Gtrain,pos, with_labels=True, node_color='lightgreen', ax=ax[i])
        else:
            Gtest = nx.from_numpy_matrix(test)
            pos = nx.circular_layout(Gtest)
            nx.draw_networkx(Gtest,pos, with_labels=True, node_color='peachpuff',ax=ax[i])
        ax[i].set_axis_off()

    plt.show()

#### plot the cicular layout grapoh
def plot_graph():
    g = nx.random_geometric_graph(22, 0.2)
    pos = nx.circular_layout(g)
    nx.draw_networkx(g, pos, with_labels=True)

## convert predictions into discrete ranks
def rank_ypred(y,maxbetweenness):
    y[y <= (0.33 * maxbetweenness)] = 0
    y[(y > (0.33 * maxbetweenness)) & (y <= (0.66 * maxbetweenness))] = 1
    y[y > (0.66 * maxbetweenness)] = 2
    return y

def rank_yegr(y,V):
        y[y <= (0.25 * V)] = 0
        y[(y > (0.25 * V)) & (y <= (0.75 * V))] = 1
        y[y >= (0.75 * V)] = 2
        return y
### metric for validation in callbacks
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]

        y_predict = np.asarray(model.predict([X_val[0], X_val[1]]))

        # y_val = np.argmax(y_val, axis=1)
        # y_predict = np.argmax(y_predict, axis=1)

        self._data.append({
            'val_rocauc': stats.kendalltau(y_val, y_predict),
        })
        return

    def get_data(self):
        return self._data
### compute kendall tau rank for the prediction matrix
def get_kendalltau(ytest, ypred):
    taulist=[]
    for count in range(ytest.shape[0]):
        tau, p_value = stats.kendalltau(ytest[count, :], ypred[count, :])
        taulist.append(tau)
    return np.mean(np.array(taulist))

### performance metrics for rank in multiple graph framework
def compute_topkperf_multigraph(y_test, y_pred):
    accurac=np.zeros((y_test.shape[0]))
    f1score=np.zeros((y_test.shape[0]))
    V = y_test.shape[1]
    for countrow in range(y_test.shape[0]):
        rank_test = rankdata(y_test[countrow,:], method='min')
        rank_pred = rankdata(y_pred[countrow,:], method='min')
        rank_test = [1 if ind >= 0.9 * V else 0 for ind in rank_test]
        rank_pred = [1 if ind >= 0.9 * V else 0 for ind in rank_pred]
        accurac[countrow] = accuracy_score(rank_test, rank_pred)
        f1score[countrow] = f1_score(rank_test, rank_pred)

    return accurac, f1score

## visualize the comparison plots of degre  betweenness , egr
def visualize_corregr(g, egr):
    btw = []
    degree = []
    nodes = []

    for node, value in nx.betweenness_centrality(g).items():
        print(node, value)
        btw.append(value)
        degree.append(g.degree(node))
        nodes.append(node)
    plt.style.use('dark_background')
    plt.plot(nodes, degree, color="dodgerblue")
    plt.plot(nodes, btw, color="green")
    plt.plot(nodes, egr, color="coral")
    print(np.corrcoef(degree, egr))

### performance metrics for rank in graph
def compute_topkperf(y_test, y_pred, margin):
    V = y_test.shape[0]
    rank_test = rankdata(y_test, method='min')
    rank_pred = rankdata(y_pred, method='min')
    rank_test = [1 if ind >= margin * V else 0 for ind in rank_test]
    rank_pred = [1 if ind >= margin * V else 0 for ind in rank_pred]
    accuracy= accuracy_score(rank_test, rank_pred)
    f1score = f1_score(rank_test, rank_pred)

    return accuracy, f1score