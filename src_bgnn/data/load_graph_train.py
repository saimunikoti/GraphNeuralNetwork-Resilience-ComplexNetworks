import dgl
import torch as th
import dgl
from dgl.data import DGLDataset
import torch
import networkx as nx
from src_bgnn.data import config as cnf
import random
import pickle
import numpy as np

class PLCgraphDataset(DGLDataset):

    def __init__(self, train_size, val_size):
        super().__init__(name='PLC graph')
        # print("train size", self.train_size)

    def process(self):

        self.train_size = 300

        self.val_size = 30

        #===== load saved dgl graph from pickle format ===========
        # filepath = cnf.datapath + "\\pubmed_weighted.pickle"
        # with open(filepath, 'rb') as f:
        #     self.graph = pickle.load(f)

        #===== load saved networkx graph in gpickle/ ===========
        filepath = cnf.datapath + "\\ppi_weighted.gpickle"
        g = nx.read_gpickle(filepath)
        g = nx.to_directed(g)

        # self.graph = dgl.from_networkx(g, node_attrs=['feature','label', 'meanfeature', 'varfeature'], edge_attrs=['weight'])
        self.graph = dgl.from_networkx(g, node_attrs=['feature', 'label'], edge_attrs=['weight'])

        #===== load original dgl graph  ===========
        # dataset = dgl.data.CoraGraphDataset()
        # self.graph = dataset[0]

        ## for VARIANCE

        # for (u,v) in g.edges():
        #     g[u][v]['weight'] = g[u][v]['weight']*g[u][v]['weight']

        self.graph.ndata['feat'] = self.graph.ndata['feature']

        # varfeat = torch.normal(0, 0.0121, size=(self.graph.ndata['feature'].shape[0], self.graph.ndata['feature'].shape[1]))
        # self.graph.ndata['feat'] = torch.add(self.graph.ndata['feature'], varfeat)

        # self.graph.ndata['varfeat'] = self.graph.ndata['feature']
        # self.graph.ndata['label'] = self.graph.ndata['label']
        self.graph.ndata['label'] = self.graph.ndata['label'].long()
        # generate mask for training nodes
        n_classes = self.num_classes

        array_lst_class = []
        for cclass in range(n_classes):
            array_lst_class.append([])

        # lst_c1 = []
        # lst_c2 = []
        # lst_c3 = []
        # lst_c4 = []
        # lst_c5 = []
        # lst_c6 = []
        # lst_c7 = []

        for node in self.graph.nodes():

            temp = self.graph.ndata['label'][node]
            array_lst_class[temp].append(node.item())

            # if self.graph.ndata['label'][node]==0:
            #     array_lst_class.append(node.item())
            #
            # elif self.graph.ndata['label'][node]==1:
            #     array_lst_class.append(node.item())
            #
            # elif self.graph.ndata['label'][node]==2:
            #     lst_c3.append(node.item())

            # elif self.graph.ndata['label'][node]==3:
            #     lst_c4.append(node.item())
            #
            # elif self.graph.ndata['label'][node]==4:
            #     lst_c5.append(node.item())
            #
            # elif self.graph.ndata['label'][node]==5:
            #     lst_c6.append(node.item())

            # elif self.graph.ndata['label'][node]==6:
            #     lst_c7.append(node.item())

        #segregate node masks

        # n_classes = 7
        # string_keys = np.arange(0,n_classes)
        # string_keys = [str(ind) for ind in string_keys]
        #
        # dict_lst = dict.fromkeys(string_keys, np.array([]))
        #
        # for node in self.graph.nodes():
        #
        #     for count_class in range(n_classes):
        #        if self.graph.ndata['label'][node] == count_class:
        #             str_key = str(count_class)
        #             dict_lst[str_key] = np.concatenate( dict_lst[str_key], [node.item()])
        #             break

        rng = random.Random(69)

        # ======== train data ==================

        array_lst_train = []
        for ind in range(n_classes):
            try:
                array_lst_train.append(rng.sample(array_lst_class[ind], self.train_size))
            except:
                array_lst_train.append(rng.sample(array_lst_class[ind], int( len(array_lst_class[ind])*0.8) ))

        lst_train = [item for sublist in array_lst_train for item in sublist]

        # ======== validation data ==================

        array_lst_avail_val = []
        for ind in range(n_classes):
            array_lst_avail_val.append([elem for elem in array_lst_class[ind] if elem not in array_lst_train[ind]])

        array_lst_val = []
        for ind in range(n_classes):
            try:
                array_lst_val.append(rng.sample(array_lst_avail_val[ind], self.val_size))
            except:
                array_lst_val.append(rng.sample(array_lst_avail_val[ind], int( len(array_lst_avail_val[ind])*0.5) ))

        lst_val = [item for sublist in array_lst_val for item in sublist]

        n_nodes = self.graph.num_nodes()

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[lst_train] = True
        val_mask[lst_val] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 121

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

def load_plcgraph(n_train, n_val):
    # load PLC data
    data = PLCgraphDataset(train_size=n_train, val_size=n_val)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    return g, data.num_classes

# CHNAGES
def inductive_split(g):

    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""

    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])

    # lst_c1 = []
    # lst_c2 = []
    # lst_c3 = []

    # for node in train_g.nodes():
    #
    #     if train_g.ndata['label'][node] == 0:
    #         lst_c1.append(node.item())
    #
    #     elif train_g.ndata['label'][node] == 1:
    #         lst_c2.append(node.item())
    #
    #     elif train_g.ndata['label'][node] == 2:
    #         lst_c3.append(node.item())

    # print("actual train nodes ", len(lst_c1), len(lst_c2), len(lst_c3))

    return train_g, val_g
