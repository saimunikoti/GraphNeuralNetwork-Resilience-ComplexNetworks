import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
import tqdm
import traceback

#### Neighbor sampler

class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds)) #This seed is the batchsampler in the dataloader at first, and the ids of the graph are yeild one by one according to the batch size
        blocks = []
        for fanout in self.fanouts: #[10,25]
            # For each seed node, sample ``fanout`` neighbors. The sampler here is newly added in v0.4.3
            frontier = dgl.sampling.sample_neighbors(g, seeds, fanout, replace=True) #10,000 edges obtained by using 1,000 seeds? node
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds) # to_black The operation is to convert the sampled subgraph into a bipartite graph suitable for calculation. The special place here is that the id in block.srcdata contains dstnodeid
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            # The length of a seed is 1000, which is the index of a batch, 1000 each batch, sample 10 neighbors, get 9640 points on 10000 sides, and then sample 25 points, get 241000 edges, 105693 points, there are two in Blocks Subgraph
            blocks.insert(0, block)
        return blocks

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x #The x input in the first round is the second-order adjacent point after sampling twice, the dimension is 10w+*602, in fact, the original label of g.ndata
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the, two layers of SAGEConv correspond to two blocks respectively
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D), blocks0 is a bipartite graph of left 9640 and right 10w+, block1 is a bipartite graph of left 1000 and right 9640
            h_dst = h[:block.number_of_dst_nodes()] # The node of each order contains its dst node at the top of the sequence, which is convenient for calculation. But how is this sampled? It is stated in the dgl.to_blockd function definition. .
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst)) #block is the graph in dglnn.SAGEConv().forward(graph,feat),feat=(h,h_dst), h is the starting node feature of 10w, h_dst is the destination node feature
            if l != len(self.layers) - 1: #When the convergence mode is mean, SAGEConv is implemented, sending all the characteristics of h to the dst node, averaging according to the dst node, adding the original characteristics of the dst node, and then outputting the new characteristics of the dst node by an fc layer. If If it is gcn, it is basically the same as mean. For details, you can see the definition of the four aggregate functions in dglnn.SAGEConv. The so-called parameter weights that need to be learned to define graphsage are the weights inside SAGEConv. For example, the dimensionality of 602 is converted to the weight of 41. .
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
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
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y
        return y

def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.

    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_mask, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_mask], labels[val_mask])

def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

#### Entry point
def run(args, device, data):
    # Unpack data, in_feats=602 ,nodes=232965 ,edges=114848857,n_classes=41, train_nid 13w training sample id
    train_mask, val_mask, in_feats, labels, n_classes, g = data
    train_nid = th.LongTensor(np.nonzero(train_mask)[0]) #np.nonzeros() returns a tuple (respectively describe the position of non-zero elements in two dimensions)
    val_nid = th.LongTensor(np.nonzero(val_mask)[0])
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)

    # Create sampler initialization, the default fanout is 10, 25, this means that the first order draws 10 times, the second order draws 25 times
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks, train-id is 15w data index, batch=1000, sampler sampler,
    dataloader = DataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks, #When the sample is not divisible by batch, the processing function needed, here is actually a method of sampling 1000 seed ids and returning the block bipartite graph
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer, input dimension 602, hidden layer 16, n_classes = 41
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, blocks in enumerate(dataloader):
            tic_step = time.time()

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID] #block0 is a bipartite graph, that is, the left side is 9640 and the right side is 105693 nodes (the number of samples will change each time!!), the side is a bipartite graph of 2410000,
            seeds = blocks[-1].dstdata[dgl.NID] #seed is 1000 seed points, the first-order sampling is 10 edges, and the bipartite block1 of 1000-9640 is obtained, and the 25 edges are sampled by 9640, and the bipartite block0 of 9640-105693 is obtained.

            # Load the input features as well as output labels, here is similar to the 105693*603 matrix of the two hops as the output, and the final output is 1000 points.
            batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device) #input_nodes is the id of the second-order point, batch_inputs is the feature corresponding to the second-order point

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(model, g, g.ndata['features'], labels, val_mask, args.batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=1)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    args = argparser.parse_args()
    
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load reddit data
    data = RedditDataset(self_loop=True)
    train_mask = data.train_mask
    val_mask = data.val_mask
    features = th.Tensor(data.features)
    in_feats = features.shape[1]
    labels = th.LongTensor(data.labels)
    n_classes = data.num_labels

    # Construct graph
    g = dgl.graph(data.graph.all_edges())
    g.ndata['features'] = features
    prepare_mp(g)

    # Pack data
    data = train_mask, val_mask, in_feats, labels, n_classes, g

    run(args, device, data)

