import dgl
import torch as th
import dgl
from dgl.data import DGLDataset
import torch
import networkx as nx
from src_bgnn.data import config as cnf
import random
import numpy as np

class PLCgraphDataset(DGLDataset):

    def __init__(self):
        super().__init__(name='PLC graph')
        self.train_size = 20
        self.val_size = 20
        # print("train size", self.train_size)

    def process(self):
        self.train_size = 3000
        self.val_size = 300
        self.test_size = 700

        filepath = cnf.datapath + "\\pubmed_weighted.gpickle"

        g = nx.read_gpickle(filepath)
        g = nx.to_directed(g)

        ## for VARIANCE
        # for (u,v) in g.edges():
        #     g[u][v]['weight'] = g[u][v]['weight']*g[u][v]['weight']

        # self.graph = dgl.from_networkx(g, node_attrs=['feature','label', 'meanfeature', 'varfeature'], edge_attrs=['weight'])

        self.graph = dgl.from_networkx(g, node_attrs=['feature','label'], edge_attrs=['weight'])

        # self.graph.ndata['feat'] = self.graph.ndata['feature']

        varfeat = torch.normal(0, np.sqrt(0.0002), size=(self.graph.ndata['feature'].shape[0], self.graph.ndata['feature'].shape[1]))
        self.graph.ndata['feat'] = torch.add(self.graph.ndata['feature'], varfeat)

        # self.graph.ndata['varfeat'] = self.graph.ndata['feature']
        # self.graph.ndata['label'] = self.graph.ndata['label']
        self.graph.ndata['label'] = self.graph.ndata['label']

        # generate mask for training nodes
        lst_c1 = []
        lst_c2 = []
        lst_c3 = []

        for node in self.graph.nodes():

            if self.graph.ndata['label'][node] == 0:
                lst_c1.append(node.item())

            elif self.graph.ndata['label'][node] == 1:
                lst_c2.append(node.item())

            elif self.graph.ndata['label'][node] == 2:
                lst_c3.append(node.item())

        rng = random.Random(69)

        lst_c1_train = rng.sample(lst_c1, self.train_size)
        lst_c2_train = rng.sample(lst_c2, self.train_size)
        lst_c3_train = rng.sample(lst_c3, self.train_size)

        lst_c1_val = [elem for elem in lst_c1 if elem not in lst_c1_train]
        lst_c2_val = [elem for elem in lst_c2 if elem not in lst_c2_train]
        lst_c3_val = [elem for elem in lst_c3 if elem not in lst_c3_train]

        lst_c1_val = rng.sample(lst_c1_val, self.val_size)
        lst_c2_val = rng.sample(lst_c2_val, self.val_size)
        lst_c3_val = rng.sample(lst_c3_val, self.val_size)

        lst_c1_test = [elem for elem in lst_c1 if elem not in lst_c1_train and lst_c1_val]
        lst_c2_test = [elem for elem in lst_c2 if elem not in lst_c2_train and lst_c2_val]
        lst_c3_test = [elem for elem in lst_c3 if elem not in lst_c3_train and lst_c3_val]

        lst_c1_test = rng.sample(lst_c1_test, self.test_size)
        lst_c2_test = rng.sample(lst_c2_test, self.test_size)
        lst_c3_test = rng.sample(lst_c3_test, self.test_size)
        lst_test = lst_c1_test + lst_c2_test + lst_c3_test
        print("no of test nodes", len(lst_test))
        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = self.graph.num_nodes()
        # n_train = int(n_nodes * 0.01)
        # n_val = int(n_nodes * 0.01)

        # test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        # val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        # test2_mask = torch.zeros(n_nodes, dtype=torch.bool)

        # train_mask[:n_train] = True
        test_mask[lst_test] = True
        # val_mask[n_train:n_train + n_val] = True
        # val_mask[lst_val] = True

        # test_mask[n_train + n_val:] = True
        # self.graph.ndata['train_mask'] = train_mask
        # self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 3

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
