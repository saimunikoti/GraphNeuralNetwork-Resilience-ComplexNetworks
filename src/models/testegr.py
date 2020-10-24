class GenEgrData1():

    def __init__(self):
        print("Generation of EGR class is invoked")

    def get_egr(self, graph):
        eig = nx.linalg.spectrum.laplacian_spectrum(graph)
        try:
            eig1 = [1/num for num in eig if num > 5e-10]
            # eig = [1 / num for num in eig[1:] if num != 0]
            eig = sum(np.abs(eig1))
        except:
            print("zero encountered in Laplacian eigen values")
        Rg = (2 / (len(graph.nodes()) - 1))*eig
        return np.round(Rg, 3)
    ### get rank of edges from eigen values of laplacian method- exact method

    def get_egrlinkrank(self, g):

        egr_new = np.zeros(len(g.edges))
        for countedge, (v1, v2) in enumerate(g.edges()):
            g[v1][v2]['edgepos'] = countedge
            gcopy = g.copy()
            gcopy.remove_edge(v1, v2)
            egr_new[countedge] = self.get_egr(gcopy)

        # egr_diff = egr_new - egr_old
        ### rank of the egr_new
        order = egr_new.argsort()
        ranks = order.argsort() + 1  # getting ranks from 1 and high rank denotes high importance score

        #### normalize to 0 to 1 range
        # ranks = (ranks - min(ranks)) / (max(ranks) - min(ranks))

        return ranks

    def get_egrnoderank(self, g):

        egr_new = np.zeros(len(g.nodes))
        for countnode, node in enumerate(g.nodes()):
            # g[v1][v2]['edgepos'] = countedge
            gcopy = g.copy()
            ## node removal at secondary
            # noderemoval=[node]
            # for neighbor in g.neighbors(node):
            #     if g.degree(neighbor) == 1:
            #         noderemoval.append(neighbor)

            gcopy.remove_node(node)
            egr_new[countnode] = self.get_egr(gcopy)
            print("node iter:", countnode)

        ranks = rankdata(egr_new, method='min')

        return ranks

    def gen_egr_plmodel(self, n, V, datadir, predictor, genflag=0):
        if genflag ==1:
            inputtensor = []
            targetvec = []
            featurevec = []
            graphvec = []

            enc = OneHotEncoder(handle_unknown='ignore')
            def get_sample(predictor):
                g = nx.generators.random_graphs.powerlaw_cluster_graph(V, 1, 0.1)

                if predictor == "adjacency":
                    temp_input = nx.adj_matrix(g).todense()
                else:
                    temp_input = nx.laplacian_matrix(g).todense()
                    temp_input = np.linalg.pinv(temp_input)

                # link rank on basis of egr
                # ranks = self.get_egrlinkrank(g)
                ## node rank on the basis of egr
                ranks = self.get_egrnoderank(g)

                # ranks = np.array([2 if ind >= int(0.75*V) else 1 if (ind > int(0.25*V)) and (ind < int(0.75*V)) else 0 for ind in temp_ranks])
                ranks = (ranks - min(ranks)) / (max(ranks) - min(ranks))

                # ranks = np.array([1/(1+np.exp(-1*ind)) for ind in temp_ranks])

                # input node feature with degree

                degfeat = (np.sum(nx.adj_matrix(g).todense(), axis=1))

                x = np.reshape(np.arange(V), (V, 1))
                Idenfeat = enc.fit_transform(x).toarray()

                feat = np.concatenate((Idenfeat, degfeat), axis=1)
                return temp_input, ranks, feat, g

            for i in range(n):
                input, target, feature,g = get_sample(predictor)
                inputtensor.append(input)
                targetvec.append(target)
                featurevec.append(feature)
                graphvec.append(g)

            # np.save(datadir+'predictor.npy', inputtensor)
            # np.save(datadir+'target.npy', targetvec)
            # np.save(datadir+'feature.npy', featurevec)

        else:
            inputtensor = np.load(datadir + 'predictor.npy')
            targetvec = np.load(datadir + 'target.npy')
            featurevec = np.load(datadir + 'feature.npy')
            graphvec = np.load(datadir + 'graphvec.npy')

        return np.array(inputtensor), np.array(targetvec), np.array(featurevec), graphvec

    def gen_egr_abmodel(self, n, V, datadir, predictor, genflag=0):
        if genflag ==1:
            inputtensor = []
            targetvec = []
            featurevec = []
            enc = OneHotEncoder(handle_unknown='ignore')
            def get_sample(predictor):
                g = nx.barabasi_albert_graph(V, 3)

                temp_input = nx.adj_matrix(g).todense()

                # link rank on basis of egr
                # ranks = self.get_egrlinkrank(g)
                ## node rank on the basis of egr
                ranks = self.get_egrnoderank(g)

                # ranks = np.array([2 if ind >= int(0.75*V) else 1 if (ind > int(0.25*V)) and (ind < int(0.75*V)) else 0 for ind in temp_ranks])
                ranks = (ranks - min(ranks)) / (max(ranks) - min(ranks))

                # ranks = np.array([1/(1+np.exp(-1*ind)) for ind in temp_ranks])

                # input node feature with degree

                degfeat = (np.sum(nx.adj_matrix(g).todense(), axis=1))

                x = np.reshape(np.arange(V), (V, 1))
                Idenfeat = enc.fit_transform(x).toarray()

                feat = np.concatenate((Idenfeat, degfeat), axis=1)
                return temp_input, ranks, feat

            for i in range(n):
                input, target, feature = get_sample(predictor)
                inputtensor.append(input)
                targetvec.append(target)
                featurevec.append(feature)

            np.save(datadir+'predictor.npy', inputtensor)
            np.save(datadir+'target.npy', targetvec)
            np.save(datadir+'feature.npy', featurevec)

        else:
            inputtensor = np.load(datadir + 'predictor.npy')
            targetvec = np.load(datadir + 'target.npy')
            featurevec = np.load(datadir + 'feature.npy')

        return np.array(inputtensor), np.array(targetvec), np.array(featurevec)

    def gen_graphegr_plmodel(self, n, V, datadir, predictor, genflag=0):
        if genflag ==1:
            targetvec = []
            # featurevec = []
            graphvec = []

            enc = OneHotEncoder(handle_unknown='ignore')
            def get_sample(predictor):

                g = nx.generators.random_graphs.powerlaw_cluster_graph(V, 1, 0.1)

                ranks = self.get_egrnoderank(g)

                ranks = (ranks - min(ranks)) / (max(ranks) - min(ranks))

                return ranks, g

            for i in range(n):
                target, g = get_sample(predictor)

                targetvec.append(target)

                graphvec.append(g)

            np.save(datadir+'target.npy', targetvec)

        else:

            targetvec = np.load(datadir + 'target.npy')


        return np.array(targetvec), graphvec

    def split_data(self, data, label, feature):
        trainingind = int(0.90*len(data))

        xtrain = data[0:trainingind,:]
        ytrain = label[0:trainingind,:]
        ftrain = feature[0:trainingind,:]

        xtest = data[trainingind:len(data),:]
        ytest = label[trainingind:len(data),:]
        ftest = feature[trainingind:len(data), :]

        return xtrain, ytrain, ftrain, xtest, ytest, ftest