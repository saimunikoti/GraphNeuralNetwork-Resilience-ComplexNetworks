from stellargraph.layer.graphsage_bayesian import GraphSAGE, MaxPoolingAggregator, MeanAggregator, MeanAggregatorvariance
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from src_bgnn.data import config as cn

## generate variance data for graph

for node in g.nodes(data=True):
    node[1]['feature'] = np.round(np.random.uniform(0.01,0.05,1433), 3)

G = StellarGraph.from_networkx(gobs, node_features="feature")
print(G.info())

batch_size = 40
# number of nodes to consider for each hop
num_samples = [15, 10, 5]
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

train_subjects, test_subjects = model_selection.train_test_split(targetdf, train_size=0.01, test_size=None)
train_targets = np.array(train_subjects)
test_targets = np.array(test_subjects)

# train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)  # train_subjects.index for selecting training nodes
test_gen = generator.flow(test_subjects.index, test_targets)

graphsage_model = GraphSAGE(layer_sizes=[32, 32, 16], generator=generator,
                            bias=True, aggregator= MeanAggregatorvariance,  dropout=0.1)

## get weightts

filepath  = cn.Uncertprop_path + fileext

model = load_model(filepath, custom_objects={"MeanAggregator": MeanAggregator})

model_weights = model.get_weights()
model_weights1 = [ model_weights[ind]**2 for ind in range(len(model_weights)) ]

## mean model
layer_name = 'reshape_6'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

## get layer weights
dense_squareweights = model_weights1[-2]

##
batches = 0

for x_batch, y_batch in generator.flow(test_subjects.index, test_targets):

    # variance output from graph
    x_inp_var, x_outvar = graphsage_model.in_outvar_tensors(x_batch, model_weights1)

    # variance output from dense layer
    x_outvar = np.matmul(x_outvar, dense_squareweights)
    
    # mean output from graphsage
    # x_outmean = intermediate_layer_model.predict(x_batch)

    # mean output from final dense layer
    x_outmean = model.predict(x_batch)

    batches += 1

    if batches >= 25:
        break

##
model1 = tf.keras.models.clone_model(model, input_tensors=None, clone_function=None)
model1.set_weights(model_weights1)

## get mean from each layer
#graphoutputfunction = K.function([model1.layers[0].input], [model1.layers[5].output])
#graphoutput = graphoutputfunction([x_batch])