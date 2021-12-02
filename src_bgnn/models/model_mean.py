import torch as th
import torch.nn as nn
import torch.functional as F
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm

class SAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, hidden_dim, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv_mean(in_feats, hidden_dim, aggregator_type= 'mean') )
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv_mean(hidden_dim, hidden_dim, aggregator_type='mean' ))
            self.layers.append(dglnn.SAGEConv_mean(hidden_dim, hidden_dim, aggregator_type='mean' ))
        else:
            self.layers.append(dglnn.SAGEConv_mean(in_feats, n_classes, aggregator_type='mean'))

        self.fc1 = nn.Linear(hidden_dim, n_classes)
        # self.bn1 = nn.BatchNorm1d(num_features=5)
        # self.bn2 = nn.BatchNorm1d(num_features=64)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x, g):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            batch_block_nodes = []

            for count in range(100000):
                try:
                    batch_block_nodes.append(block.dstnodes[count][0]['_ID'].item())
                except:
                    break

            # computing DEGREE OF NODES IN EACH BLOCK OF BATCH
            batch_degree = th.tensor([ g.out_degrees(enm) for enm in batch_block_nodes]).to(device='cuda:0')

            h = layer(block, h, batch_degree)
            # h = layer(block, h, batch_degree, model_weights[l*3+1], model_weights[3*l+2])

            # if l != len(self.layers) - 1:
                # h = self.bn2(h)
            # h = self.activation(h)

            h = self.dropout(h)

        # dense_weight = th.square(self.fc1.weight)
        # h = th.matmul(h, th.transpose(dense_weight, 0, 1))
        h = self.fc1(h)

        return h

    def inference(self, g, x, device, batch_size, num_workers):

        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this ?

        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.hidden_dim if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test
