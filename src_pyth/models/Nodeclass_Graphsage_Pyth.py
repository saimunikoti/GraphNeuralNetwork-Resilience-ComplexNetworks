import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from src_pyth.data import config as cnf
import torch.optim as optim
import time
import argparse

from src_pyth.models.model import SAGE
from src_pyth.data.load_graph import load_plcgraph, inductive_split
from src.data import utils as ut
from src.data import config
import pickle
import warnings
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
warnings.filterwarnings("ignore")
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluatev0(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval() # change the mode

    with th.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)

    model.train() # rechange the model mode to training

    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def evaluate(model, test_nfeat, test_labels, device, dataloader, loss_fcn):

    """
    Evaluate the model on the given data set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval() # change the mode

    test_acc = 0.0
    test_loss = 0.0
    class1acc = 0.0

    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        with th.no_grad():
            # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
            batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

            temp_pred = th.argmax(batch_pred, dim=1)
            current_acc = accuracy_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy() )
            test_acc = test_acc + ((1 / (step + 1)) * (current_acc - test_acc))

            cnfmatrix = confusion_matrix(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy())
            class1acc = class1acc + ((1 / (step + 1)) * (cnfmatrix[0][0] / np.sum(cnfmatrix[0, :]) - class1acc))

            print(cnfmatrix)

            # correct = temp_pred.eq(batch_labels)
            # test_acc = test_acc + correct

            loss = loss_fcn(batch_pred, batch_labels)
            test_loss = test_loss + ((1 / (step + 1)) * (loss.data - test_loss))

    model.train() # rechange the model mode to training

    return test_acc, test_loss, class1acc

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    th.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = th.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, valid_loss_min.item()

#### Entry point
def run(args, device, data, checkpoint_path, best_model_path, trainflag=0):

    # Unpack data

    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data

    in_feats = train_nfeat.shape[1]

    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]

    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]

    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    dataloader_device = th.device('cpu')

    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        dataloader_device = device

    # define dataloader function
    def get_dataloader(train_g, train_nid, sampler):

        dataloader = dgl.dataloading.NodeDataLoader(
            train_g,
            train_nid,
            sampler,
            device=dataloader_device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)

        return dataloader

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)

    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if trainflag == 1:

        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(',')])

        # Create PyTorch DataLoader for constructing blocks
        dataloader = get_dataloader(train_g, train_nid, sampler)

        # validata dataloader
        valdataloader = get_dataloader(val_g, val_nid, sampler)

        # Training loop
        valid_loss_min = np.Inf

        for epoch in range(args.num_epochs):

            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            # tic_step = time.time()

            model.train()
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
                batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                            seeds, input_nodes, device)
                blocks = [block.int().to(device) for block in blocks]

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # train_loss = train_loss + ((1 / (step + 1)) * (loss.data - train_loss))

            model.eval()
            train_acc, train_loss, _ = evaluate(model, train_nfeat,train_labels,device, dataloader,loss_fcn)
            val_acc, valid_loss, _ = evaluate(model, val_nfeat, val_labels,device, valdataloader, loss_fcn)

            print('Epoch: {} \tTraining acc: {:.6f} \tValidation acc: {:.6f} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch,
                train_acc,
                val_acc,
                train_loss,
                valid_loss
                ))

            checkpoint = {
                'valid_loss_min': valid_loss,
                'state_dict': model.state_dict(),
            }

            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            # TODO: save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                valid_loss_min = valid_loss

    elif trainflag == 0:

        ckp_path = cnf.modelpath + "\\current_checkpoint.pt"
        model, valid_loss_min = load_ckp(ckp_path, model, optimizer)

        print("model = ", model)
        print("optimizer = ", optimizer)
        print("valid_loss_min = ", valid_loss_min)

        # dropout and batch normalization to evaluation mode
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

        # Create PyTorch DataLoader for constructing blocks
        dataloader = get_dataloader(test_g, test_nid, sampler)

        model.eval()

        test_acc, test_loss, class1acc = evaluate(model, test_nfeat, test_labels, device, dataloader, loss_fcn)

        print('Test acc: {:.6f} \tClass1acc: {:.6f} \tTess Loss: {:.6f}'.format(test_acc, class1acc, test_loss))

    # print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='PLC')
    argparser.add_argument('--num-epochs', type=int, default= 5)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,15')
    argparser.add_argument('--batch-size', type=int, default=599)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    # argparser.add_argument('--inductive', action='store_true',
    #                        help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    filepath = cnf.datapath + "dgl" + '\\plc_600_egr' + ".gpickle"
    # changes
    if args.dataset == 'PLC':
        g, n_classes = load_plcgraph(filepath=filepath, train_ratio=0.75, valid_ratio=0.15)
    # elif args.dataset == 'reddit':
    #     g, n_classes = load_reddit()
    # elif args.dataset == 'ogbn-products':
    #     g, n_classes = load_ogb('ogbn-products')

    else:
        raise Exception('unknown dataset')

    # if args.inductive:
    train_g, val_g, test_g = inductive_split(g)

    train_nfeat = train_g.ndata.pop('features')
    val_nfeat = val_g.ndata.pop('features')
    test_nfeat = test_g.ndata.pop('features')
    train_labels = train_g.ndata.pop('labels')
    val_labels = val_g.ndata.pop('labels')
    test_labels = test_g.ndata.pop('labels')

    print("no of train, and val nodes", train_nfeat.shape, val_nfeat.shape)

    # else:
    #     train_g = val_g = test_g = g
    #     train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
    #     train_labels = val_labels = test_labels = g.ndata.pop('labels')

    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels

    run(args, device, data, cnf.modelpath + "\\current_checkpoint.pt", cnf.modelpath + "\\plc_5kv.pt", 0)

