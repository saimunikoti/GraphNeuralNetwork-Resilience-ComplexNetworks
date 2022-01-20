import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from src_pyth.data import config as cnf
import torch.optim as optim
import time
import argparse

from src_bgnn.models.model_var import SAGE
from src_bgnn.data.load_graph_test import load_plcgraph, inductive_split
from src_bgnn.data import utils as ut
from src_bgnn.data import config as cnf
import pickle
import warnings
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
warnings.filterwarnings("ignore")
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
##

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate_test(model, test_labels, device, dataloader, loss_fcn):

    """
    Evaluate the model on the given data set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval() # change the mode

    test_loss = 0.0

    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        with th.no_grad():
            # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
            batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs, g)

            temp_pred = th.argmax(batch_pred, dim=1)

            # NLL metric

            # loss = nn.NLLLoss()
            # output = loss(batch_pred, batch_labels)
            # current_acc = accuracy_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy() )
            # test_acc = test_acc + ((1 / (step + 1)) * (current_acc - test_acc))

            # cnfmatrix = confusion_matrix(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy())
            # class1acc = class1acc + ((1 / (step + 1)) * (cnfmatrix[0][0] / np.sum(cnfmatrix[0, :]) - class1acc))
            #
            # print(batch_pred[0:100])

            # correct = temp_pred.eq(batch_labels)
            # test_acc = test_acc + correct

            loss = loss_fcn(batch_pred, batch_labels)
            test_loss = test_loss + ((1 / (step + 1)) * (loss.data - test_loss))

    model.train() # rechange the model mode to training

    return test_loss

def evaluate_test_mc_old(model, test_labels, device, dataloader, loss_fcn, g, T):

    """
    Evaluate the model on the given data set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """

    def apply_dropout(m):
        if type(m) == nn.Dropout:
            m.train()

    model.eval() # change the mode
    model.apply(apply_dropout)

    test_loss = 0.0
    batch_predc1 = []
    batch_predc2 = []
    batch_predc3 = []
    mcensemble_loss = []

    for countmc in range(T):

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):

            with th.no_grad():
                # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
                batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                            seeds, input_nodes, device)

                blocks = [block.int().to(device) for block in blocks]

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs, g)

                batch_predc1.append(batch_pred.cpu().detach().numpy()[0][0])
                batch_predc2.append(batch_pred.cpu().detach().numpy()[0][1])
                batch_predc3.append(batch_pred.cpu().detach().numpy()[0][2])

                loss = loss_fcn(batch_pred, batch_labels)
                mcensemble_loss.append(loss.cpu().detach().numpy())
                test_loss = test_loss + ((1 / (step + 1)) * (loss.data - test_loss))

            break
        # model.train() # rechange the model mode to training
        print("mc: ", countmc)

    return batch_predc1, batch_predc2, batch_predc3, mcensemble_loss

def evaluate_test_mc(model, test_labels, device, dataloader, loss_fcn, g, n_mcsim):

    """
    Evaluate the model on the given data set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """

    def apply_dropout(m):
        if type(m) == nn.Dropout:
            m.train()

    model.eval() # change the mode
    model.apply(apply_dropout)

    onehot_encoder = OneHotEncoder(sparse=False)

    classlabellist = []
    for count in range(n_classes):
        classlabellist.append([count])

    classlabellist = np.array(classlabellist)
    onehot_encoder.fit_transform(classlabellist)

    filepath = cnf.modelpath + "Resultsdic_amazon-comp_meanpred_var12.pkl"

    with open(filepath, 'rb') as f:
        mean_dic = pickle.load(f)

    diffmeanarray = np.mean(mean_dic['diffmean_array'], axis=2)
    predmeanarray = np.mean(mean_dic['pred_array'], axis=2)

    n_testsamples = dataloader.__len__()

    sigmatot_array = np.zeros(shape=(n_testsamples, n_classes))
    nll_array = np.zeros(shape=(n_testsamples, n_classes))
    true_array = np.zeros(shape=(n_testsamples, n_classes))
    loss_array = np.zeros(shape=(n_testsamples, n_mcsim))

    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):

        var_pred = np.zeros(shape=(n_mcsim, n_classes))

        for countmc in range(n_mcsim):

            with th.no_grad():
                # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
                batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                            seeds, input_nodes, device)

                blocks = [block.int().to(device) for block in blocks]

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs, g)

                var_pred[countmc, :] = batch_pred.cpu().detach().numpy()[0]

                loss = loss_fcn(batch_pred, batch_labels)
                # tloss.append(loss.item())
                loss_array[step, countmc] = loss.item()
                # test_loss = test_loss + ((1 / (step + 1)) * (loss.data - test_loss))

            print("node-mcit", step, countmc)

        var_pred = np.mean(var_pred, axis=0)

        for countc in range(n_classes):
            sigmatot_array[step, countc] = var_pred[countc] + diffmeanarray[step, countc]

        # nll for each sample, class and mc_sim
        for countc in range(n_classes):
            nll_array[step, countc] = (0.5 * np.log(sigmatot_array[step, countc])) + \
                                    (1 / (2 * sigmatot_array[step, countc]) * \
                                    np.square(mean_dic['true_array'][step, countc] - predmeanarray[step, countc] ))

        temp_label = np.reshape(batch_labels.cpu().detach().numpy(), (1, 1))

        for countc in range(n_classes):
            true_array[step, countc] = onehot_encoder.transform(temp_label)[0][countc]

    filepath = cnf.modelpath + "Resultsdic_amazon-comp_varpred_var12.pkl"

    mean_dic['sigmatot_array'] = sigmatot_array
    mean_dic['nll_array'] = nll_array
    mean_dic['varloss_array'] = loss_array

    with open(filepath, 'wb') as f:
        pickle.dump(mean_dic, f)

    # average results across monte carlo simulations and save in csv
    Resultsdf = pd.DataFrame()

    for count in range(n_classes):
        predname = 'pred_array_c' + str(count+1)
        truename = 'true_array_c' + str(count+1)
        propvarname = 'prop_var_c' + str(count+1)

        Resultsdf[predname] = np.mean(mean_dic['pred_array'][:,count,:], axis=1)
        Resultsdf[truename] = mean_dic['true_array'][:,count]
        Resultsdf[propvarname] = mean_dic['sigmatot_array'][:,count]

    filepath = cnf.modelpath + "Resultsdic_amazon-comp_var12.csv"

    Resultsdf.to_csv(filepath)

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

def run(args, device, data, best_model_path):

    # Unpack data
    n_classes, test_g, test_nfeat, test_labels, g = data

    in_feats = test_nfeat.shape[1]

    test_nid = th.nonzero(test_g.ndata['test_mask'], as_tuple=True)[0]

    # val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]

    # test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
    # test2_nid = th.nonzero(test2_g.ndata['test2_mask'], as_tuple=True)[0]

    dataloader_device = th.device('cpu')
    if args.sample_gpu:
        test_nid = test_nid.to(device)
        # copy only the csc to the GPU
        test_g = test_g.formats(['csc'])
        test_g = test_g.to(device)
        dataloader_device = device

    # define dataloader function
    def get_dataloader(test_g, test_nid, sampler):

        dataloader = dgl.dataloading.NodeDataLoader(
            test_g,
            test_nid,
            sampler,
            device=dataloader_device,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)

        return dataloader

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)

    model = model.to(device)

    # weights = [17, 1]
    # class_weights = th.FloatTensor(weights).to(device)
    loss_fcn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model, valid_loss_min = load_ckp(best_model_path, model, optimizer)

    print("model = ", model)
    print("optimizer = ", optimizer)
    print("valid_loss_min = ", valid_loss_min)

    # dropout and batch normalization to evaluation mode
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    # Create PyTorch DataLoader for constructing blocks
    dataloader = get_dataloader(test_g, test_nid, sampler)

    model.eval()

    # test_loss = evaluate_test(model, test_labels, device, dataloader, loss_fcn)

    evaluate_test_mc(model, test_labels, device, dataloader, loss_fcn, g, 100)

    # print('Tess Loss: {:.6f}'.format(mcensemble_loss))

    # Resultsdf = pd.DataFrame(
    #     columns=['mcensemble_acc', 'mcensemble_loss'])
    #
    # Resultsdf['mcensemble_predc1'] = batch_predc1
    # Resultsdf['mcensemble_predc2'] = batch_predc2
    # Resultsdf['mcensemble_predc3'] = batch_predc3
    #
    # Resultsdf['mcensemble_loss'] = mcensemble_loss
    #
    # filepath = cnf.modelpath + "Resultsdf_mcensemblevar_s1c2_var2.xlsx"
    # Resultsdf.to_excel(filepath, index=False)

    # variance of predictions

    # print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    th.manual_seed(42)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='PLC')
    argparser.add_argument('--num-epochs', type=int, default= 100)
    argparser.add_argument('--num_hidden', type=int, default=48)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='8,10')
    argparser.add_argument('--batch-size', type=int, default=1)
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

    # changes
    if args.dataset == 'PLC':
        g, n_classes = load_plcgraph()
    else:
        raise Exception('unknown dataset')

    edgelistnew = g.edges()
    # square the edge prob for variance computation
    for count in range(len(edgelistnew[0])):
        g.edata['weight'][th.tensor([count])] = th.square(g.edata['weight'][th.tensor([count])])

    # if args.inductive:
    test_g = inductive_split(g)

    # train_nfeat = train_g.ndata.pop('features')
    # val_nfeat = val_g.ndata.pop('features')
    test_nfeat = test_g.ndata.pop('features')
    # test2_nfeat = test2_g.ndata.pop('features')
    # train_labels = train_g.ndata.pop('labels')
    # val_labels = val_g.ndata.pop('labels')
    test_labels = test_g.ndata.pop('labels')
    # test2_labels = test2_g.ndata.pop('labels')

    print("test graph size :", test_nfeat.shape)

    if not args.data_cpu:
        test_nfeat = test_nfeat.to(device)
        test_labels = test_labels.to(device)

    # Pack data
    data = n_classes, test_g, test_nfeat, test_labels, g

    start_time = time.time()
    run(args, device, data, cnf.modelpath + "\\amazon-comp_uc.pt")
    end_time = time.time()-start_time

    print("total time", end_time)

