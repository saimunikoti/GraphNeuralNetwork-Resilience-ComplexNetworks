from src.data import utils as ut
from src.data import config
from src.features import build_features as bf

## generate training data

graphsizelist= [100, 200, 400, 400, 500, 500, 800, 800, 1000, 1000, 2000, 2000, 4000, 4000]
# grpahsizelist= [100, 300, 600, 900, 1200, 1500, 2000, 2500, 3000]

fileext = "\egr_pl_"+ str(sum(graphsizelist))

# generate graph and labels for synthetic plcluster networks
Listgraph, Listlabel = ut.get_graphfeaturelabel_syn('plc', 'egr', graphsizelist)

fileext = "\power-US-Grid.txt"
# generate graph and corresponding labels for real world networks
g = ut.get_graphtxt(filepath)
g = g.to_undirected()
Listgraph, Listlabel = ut.get_graphfeaturelabel_real(config.datapath + 'raw'+ fileext, "egr")

# combine graphs into one disjoint union graph
g = bf.combine_graphs(Listgraph)

## save graph and corresponding labels features

nx.write_gpickle(g, config.datapath + 'processed'+fileext + ".gpickle")

with open(config.datapath + 'processed'+ fileext + "_label.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

# with open(".\data\processed" + fileext+ "_feature.pickle", 'wb') as b:
#     pickle.dump(feature, b)

## load  synthetic graphs and labels
g = nx.read_gpickle(config.datapath + 'processed\\'+ fileext+".gpickle")

with open(config.datapath + 'processed\\' + fileext+ "_label.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

## load real graphs and labels

g = nx.read_gpickle(config.datapath + 'processed\\'+ fileext+".gpickle")

with open(config.datapath + 'processed\\' + fileext+ "_wslabel.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

## renumbering of nodes for real graphs
# mdf = pd.DataFrame(data= np.array(g.nodes), columns=['data'] )
# mdf['newdata'] = np.array(g.nodes)-1
# mapping  = mdf.set_index('data')['newdata'].to_dict()
# g = nx.relabel_nodes(g, mapping)

## generate target vector for syn graph

targetdf = bf.getgraphtargetdf(Listlabel, g)
targetdf.loc[targetdf.metric==0,'metric'] = 0.001

category = pd.cut(targetdf.metric, bins=[0,0.60,0.7,1.0],labels=[0,1,2])
targetdf['metric'] = category
targetdf = pd.get_dummies(targetdf.metric)

### feature vector
for node_id, node_data in g.nodes(data=True):
    node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1,1]
    # node_data["feature"] = [g.degree(node_id), 1, 1,1]



##

