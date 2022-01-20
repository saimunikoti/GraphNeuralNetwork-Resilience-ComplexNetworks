import dgl
import torch as th
import dgl
from dgl.data import DGLDataset
import torch
import networkx as nx
from src_bgnn.data import config as cnf
import random
import numpy as np
import pickle
import copy

class PLCgraphDataset(DGLDataset):

    def __init__(self):
        super().__init__(name='PLC graph')
        self.train_size = 200
        self.val_size = 10
        # print("train size", self.train_size)

    def process(self):
        self.train_size = 300
        self.val_size = 30
        self.test_size = 100

        #===== load saved dgl graph from pickle format ===========
        # filepath = cnf.datapath + "\\pubmed_weighted.pickle"
        # with open(filepath, 'rb') as f:
        #     self.graph = pickle.load(f)

        #===== load saved networkx graph in gpickle/ ===========
        filepath = cnf.datapath + "\\amazon_computer_weighted.gpickle"
        g = nx.read_gpickle(filepath)
        g = nx.to_directed(g)

        # self.graph = dgl.from_networkx(g, node_attrs=['feature','label', 'meanfeature', 'varfeature'], edge_attrs=['weight'])
        self.graph = dgl.from_networkx(g, node_attrs=['feature','label'], edge_attrs=['weight'])

        #===== load original dgl graph  ===========
        # dataset = dgl.data.CoraGraphDataset()
        # self.graph = dataset[0]

        # g = nx.read_gpickle(filepath)
        # g = nx.to_directed(g)

        ## for VARIANCE
        # for (u,v) in g.edges():
        #     g[u][v]['weight'] = g[u][v]['weight']*g[u][v]['weight']

        # self.graph = dgl.from_networkx(g, node_attrs=['feature','label', 'meanfeature', 'varfeature'], edge_attrs=['weight'])

        # self.graph = dgl.from_networkx(g, node_attrs=['feature','label'], edge_attrs=['weight'])

        # self.graph.ndata['feat'] = self.graph.ndata['feature']

        varfeat = torch.normal(0, np.sqrt(0.042), size=(self.graph.ndata['feature'].shape[0], self.graph.ndata['feature'].shape[1]))
        self.graph.ndata['feat'] = torch.add(self.graph.ndata['feature'], varfeat)

        # self.graph.ndata['varfeat'] = self.graph.ndata['feature']
        # self.graph.ndata['label'] = self.graph.ndata['label']
        self.graph.ndata['label'] = self.graph.ndata['label'].long()

        # generate mask for training nodes

        n_classes = self.num_classes

        array_lst_class = []

        for cclass in range(n_classes):
            array_lst_class.append([])

        for node in self.graph.nodes():
            temp = self.graph.ndata['label'][node]
            array_lst_class[temp].append(node.item())


        rng = random.Random(69)

        array_lst_train = []
        for ind in range(n_classes):
            try:
                array_lst_train.append(rng.sample(array_lst_class[ind], self.train_size))
            except:
                array_lst_train.append(rng.sample(array_lst_class[ind], int( len(array_lst_class[ind])*0.8) ))

        #=================== val data ======================

        array_lst_avail_val = []
        for ind in range(n_classes):
            array_lst_avail_val.append([elem for elem in array_lst_class[ind] if elem not in array_lst_train[ind]])

        array_lst_val = []
        for ind in range(n_classes):
            try:
                array_lst_val.append(rng.sample(array_lst_avail_val[ind], self.val_size))
            except:
                array_lst_val.append(rng.sample(array_lst_avail_val[ind], int( len(array_lst_avail_val[ind])*0.5) ))

        # ================ test date =====================

        array_lst_avail_test = []
        for ind in range(n_classes):
            array_lst_avail_test.append([elem for elem in array_lst_class[ind] if elem not in array_lst_train[ind] and elem not in array_lst_val[ind] ])

        array_lst_test = []
        for ind in range(n_classes):
            try:
                array_lst_test.append(rng.sample(array_lst_avail_test[ind], self.test_size))
            except:
                array_lst_test.append(rng.sample(array_lst_avail_test[ind], int( len(array_lst_avail_test[ind])*1.0)))

        lst_test = [item for sublist in array_lst_test for item in sublist]

        print("no of test nodes", len(lst_test))
        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = self.graph.num_nodes()

        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        test_mask[lst_test] = True

        self.graph.ndata['test_mask'] = test_mask

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 10

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

def load_plcgraph():
    # load PLC data
    data = PLCgraphDataset()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    return g, data.num_classes

# CHNAGES

def inductive_split(g):

    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""

    # train_g = g.subgraph(g.ndata['train_mask'])
    # val_g = g.subgraph(g.ndata['val_mask'])
    # test_g = g.subgraph(g.ndata['test_mask'])
    # val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return test_g
