import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from src_pyth.data import config as cnf
import torch.optim as optim
import time
import argparse

from src_bgnn.models.model_mean import SAGE
# from src_bgnn.models.model import SAGE
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

def evaluate_test(model, test_labels, device, dataloader, loss_fcn, g):

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
            batch_pred = model(blocks, batch_inputs, g)

            temp_pred = th.argmax(batch_pred, dim=1)

            current_acc = accuracy_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy())
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

# def evaluate_test_mc_backup(model, test_labels, device, dataloader, loss_fcn, g, n_mcsim):
#
#     """
#     Evaluate the model on the given data set specified by ``val_nid``.
#     g : The entire graph.
#     inputs : The features of all the nodes.
#     labels : The labels of all the nodes.
#     val_nid : the node Ids for validation.
#     device : The GPU device to evaluate on.
#     """
#
#     def apply_dropout(m):
#         if type(m) == nn.Dropout:
#             m.train()
#
#     model.eval() # change the mode
#     model.apply(apply_dropout)
#
#     m_softmax = nn.Softmax(dim=1)
#
#     onehot_encoder = OneHotEncoder(sparse=False)
#     onehot_encoder.fit_transform(np.array([[0],[1],[2],[3],[4],[5],[6]]))
#
#     Resultsdf = pd.DataFrame()
#
#     predc1_list = []
#     predc2_list = []
#     predc3_list = []
#     predc4_list = []
#     predc5_list = []
#     predc6_list = []
#     predc7_list = []
#
#     diffMean_list_c1 = []
#     diffMean_list_c2 = []
#     diffMean_list_c3 = []
#     diffMean_list_c4 = []
#     diffMean_list_c5 = []
#     diffMean_list_c6 = []
#     diffMean_list_c7 = []
#
#     truec1_list = []
#     truec2_list = []
#     truec3_list = []
#     truec4_list = []
#     truec5_list = []
#     truec6_list = []
#     truec7_list = []
#
#     loss_list = []
#
#     for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
#         predc1 = []
#         predc2 = []
#         predc3 = []
#         predc4 = []
#         predc5 = []
#         predc6 = []
#         predc7 = []
#
#         tloss  = []
#
#         for countmc in range(n_mcsim):
#
#             with th.no_grad():
#                 # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
#                 batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
#                                                             seeds, input_nodes, device)
#
#                 blocks = [block.int().to(device) for block in blocks]
#
#                 # Compute loss and prediction
#                 batch_pred = model(blocks, batch_inputs, g)
#
#                 pred_prob = m_softmax(batch_pred)
#
#                 predc1.append(pred_prob.cpu().detach().numpy()[0][0])
#                 predc2.append(pred_prob.cpu().detach().numpy()[0][1])
#                 predc3.append(pred_prob.cpu().detach().numpy()[0][2])
#                 predc4.append(pred_prob.cpu().detach().numpy()[0][3])
#                 predc5.append(pred_prob.cpu().detach().numpy()[0][4])
#                 predc6.append(pred_prob.cpu().detach().numpy()[0][5])
#                 predc7.append(pred_prob.cpu().detach().numpy()[0][6])
#
#                 # temp_pred = th.argmax(batch_pred, dim=1)
#
#                 # current_acc = accuracy_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy())
#                 # test_acc = test_acc + ((1 / (step + 1)) * (current_acc - test_acc))
#
#                 # mcensemble_acc.append(test_acc)
#
#                 # cnfmatrix = confusion_matrix(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy())
#                 # class1acc = class1acc + ((1 / (step + 1)) * (cnfmatrix[0][0] / np.sum(cnfmatrix[0, :]) - class1acc))
#
#                 # print(cnfmatrix)
#
#                 # correct = temp_pred.eq(batch_labels)
#                 # test_acc = test_acc + correct
#
#                 loss = loss_fcn(batch_pred, batch_labels)
#                 tloss.append(loss.item())
#
#                 # test_loss = test_loss + ((1 / (step + 1)) * (loss.data - test_loss))
#
#             print("node-mciter", step, countmc)
#
#         # model.train() # rechange the model mode to training
#         predc1_list.append(np.mean(predc1))
#         predc2_list.append(np.mean(predc2))
#         predc3_list.append(np.mean(predc3))
#         predc4_list.append(np.mean(predc4))
#         predc5_list.append(np.mean(predc5))
#         predc6_list.append(np.mean(predc6))
#         predc7_list.append(np.mean(predc7))
#
#         temp_label = np.reshape(batch_labels.cpu().detach().numpy(), (1, 1))
#
#         truec1_list.append(onehot_encoder.transform(temp_label)[0][0])
#         truec2_list.append(onehot_encoder.transform(temp_label)[0][1])
#         truec3_list.append(onehot_encoder.transform(temp_label)[0][2])
#         truec4_list.append(onehot_encoder.transform(temp_label)[0][3])
#         truec5_list.append(onehot_encoder.transform(temp_label)[0][4])
#         truec6_list.append(onehot_encoder.transform(temp_label)[0][5])
#         truec7_list.append(onehot_encoder.transform(temp_label)[0][6])
#
#         diffMean_list_c1.append(np.mean(np.square(np.array(predc1)-np.mean(predc1))))
#         diffMean_list_c2.append(np.mean(np.square(np.array(predc2)-np.mean(predc2))))
#         diffMean_list_c3.append(np.mean(np.square(np.array(predc3)-np.mean(predc3))))
#         diffMean_list_c4.append(np.mean(np.square(np.array(predc4)-np.mean(predc4))))
#         diffMean_list_c5.append(np.mean(np.square(np.array(predc5)-np.mean(predc5))))
#         diffMean_list_c6.append(np.mean(np.square(np.array(predc6)-np.mean(predc6))))
#         diffMean_list_c7.append(np.mean(np.square(np.array(predc7)-np.mean(predc7))))
#
#         loss_list.append(np.mean(tloss))
#
#     Resultsdf['ypred_c1'] = predc1_list
#     Resultsdf['ypred_c2'] = predc2_list
#     Resultsdf['ypred_c3'] = predc3_list
#     Resultsdf['ypred_c4'] = predc4_list
#     Resultsdf['ypred_c5'] = predc5_list
#     Resultsdf['ypred_c6'] = predc6_list
#     Resultsdf['ypred_c7'] = predc7_list
#
#     Resultsdf['ytrue_c1'] = truec1_list
#     Resultsdf['ytrue_c2'] = truec2_list
#     Resultsdf['ytrue_c3'] = truec3_list
#     Resultsdf['ytrue_c4'] = truec4_list
#     Resultsdf['ytrue_c5'] = truec5_list
#     Resultsdf['ytrue_c6'] = truec6_list
#     Resultsdf['ytrue_c7'] = truec7_list
#
#     Resultsdf['predloss'] = loss_list
#
#     Resultsdf['diffMean_c1'] = diffMean_list_c1
#     Resultsdf['diffMean_c2'] = diffMean_list_c2
#     Resultsdf['diffMean_c3'] = diffMean_list_c3
#     Resultsdf['diffMean_c4'] = diffMean_list_c4
#     Resultsdf['diffMean_c5'] = diffMean_list_c5
#     Resultsdf['diffMean_c6'] = diffMean_list_c6
#     Resultsdf['diffMean_c7'] = diffMean_list_c7
#
#     filepath = cnf.modelpath + "Results_cora_meanpred_var25.xlsx"
#     Resultsdf.to_excel(filepath, index=False)

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

    m_softmax = nn.Softmax(dim=1)

    onehot_encoder = OneHotEncoder(sparse=False)

    classlabellist = []
    for count in range(n_classes):
        classlabellist.append([count])

    classlabellist = np.array(classlabellist)
    onehot_encoder.fit_transform(classlabellist)

    # onehot_encoder.fit_transform(np.array([[0],[1],[2],[3],[4],[5],[6]]))

    n_testsamples = dataloader.__len__()

    pred_array = np.zeros(shape=(n_testsamples, n_classes, n_mcsim))
    true_array = np.zeros(shape=(n_testsamples, n_classes))
    diffmean_array = np.zeros(shape=(n_testsamples, n_classes, n_mcsim))
    loss_array = np.zeros(shape=(n_testsamples, n_mcsim))

    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):

        for countmc in range(n_mcsim):

            with th.no_grad():
                # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
                batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                            seeds, input_nodes, device)

                blocks = [block.int().to(device) for block in blocks]

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs, g)

                pred_prob = m_softmax(batch_pred)

                for countc in range(n_classes):
                    pred_array[step, countc, countmc] = pred_prob.cpu().detach().numpy()[0][countc]

                loss = loss_fcn(batch_pred, batch_labels)

                loss_array[step, countmc] = loss.item()

            print("node-mciter", step, countmc)

        temp_label = np.reshape(batch_labels.cpu().detach().numpy(), (1, 1))

        for countc in range(n_classes):
            true_array[step, countc] = onehot_encoder.transform(temp_label)[0][countc]

        for countc in range(n_classes):
            diffmean_array[step, countc,:] = np.square(pred_array[step,countc,:]-np.mean(pred_array[step, countc,:]))

    Resultsdic = {}
    Resultsdic['pred_array'] = pred_array
    Resultsdic['true_array'] = true_array
    Resultsdic['diffmean_array'] = diffmean_array
    Resultsdic['loss_array'] = loss_array

    filepath = cnf.modelpath + "Resultsdic_amazon-comp_meanpred_var12.pkl"

    with open(filepath, 'wb') as f:
        pickle.dump(Resultsdic, f)

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

    loss_fcn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model, valid_loss_min = load_ckp(best_model_path, model, optimizer)

    # model_weights = [p for p in model.parameters() if p.requires_grad]

    print("model = ", model)
    print("optimizer = ", optimizer)
    print("valid_loss_min = ", valid_loss_min)

    # dropout and batch normalization to evaluation mode
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    # Create PyTorch DataLoader for constructing blocks
    dataloader = get_dataloader(test_g, test_nid, sampler)

    model.eval()

    # test_acc, class1acc, test_loss = evaluate_test(model, test_labels, device, dataloader, loss_fcn, g)

    evaluate_test_mc(model, test_labels, device, dataloader, loss_fcn, g, 100)

    # print('Test acc: {:.6f} \tClass1acc: {:.6f} \tTess Loss: {:.6f}'.format(test_acc, class1acc, test_loss))

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

    # if args.inductive: entire graph for test
    test_g = inductive_split(g)

    # train_nfeat = train_g.nda ta.pop('features')
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

