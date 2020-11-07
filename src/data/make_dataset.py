from src.data import utils as ut
from src.data import config

## generate training data
grpahsizelist= [100, 200, 400, 400, 500, 500, 800, 800, 1000, 1000, 2000, 2000, 4000, 4000]
# grpahsizelist= [100, 300, 600, 900, 1200, 1500, 2000, 2500, 3000]

fileext = "\egr_plc_"+ str(sum(grpahsizelist))

# generate graph and labels for synthetic plcluster networks
Listgraph, Listlabel = ut.get_graphfeaturelabel_syn('pl', 'egr', grpahsizelist)

fileext = "\power-US-Grid.txt"
# generate graph and corresponding labels for real world networks
Listgraph, Listlabel = ut.get_graphfeaturelabel_real(config.datapath + 'raw'+ fileext, "egr")

# combine graphs into one disjoint union graph
g = bf.combine_graphs(Listgraph)

## save graph and corresponding labels features
nx.write_gpickle(g, config.datapath + 'processed'+fileext + ".gpickle")

with open(".\data\processed" + fileext + "_wslabel.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

# with open(".\data\processed" + fileext+ "_feature.pickle", 'wb') as b:
#     pickle.dump(feature, b)

## load graphs and labelsl

g = nx.read_gpickle(config.datapath + 'processed\\'+ fileext+".gpickle")

with open(config.datapath + 'processed\\' + fileext+ "_label.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

# load feature
# with open(".\data\processed"+fileext + "_feature.pickle", 'rb') as b:
#     feature = pickle.load(b)

## renumbering of nodes for real graphs
mdf = pd.DataFrame(data= np.array(g.nodes), columns=['data'] )
mdf['newdata'] = np.array(g.nodes)-1
mapping  = mdf.set_index('data')['newdata'].to_dict()
g = nx.relabel_nodes(g, mapping)

## generate target vector

targetlabel = Listlabel[0]

for countlen in range(len(Listlabel)-1):
    targetlabel = np.concatenate((targetlabel, Listlabel[countlen+1]), axis=0)

targetdf = pd.DataFrame()
# label = np.reshape(label, (label.shape[1],))
targetdf['btw'] = targetlabel

### feature vector

for node_id, node_data in g.nodes(data=True):
    node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1,1]
    # node_data["feature"] = [g.degree(node_id), 1, 1,1]

##

