
=================== src.data files ==========================
1. make_dataset.py: generate node data (targets and features) for synthetic graphs 
2. make_dataset_links.py: generate links data (targets and features) for synthetic graphs 
3. make_dataset_realgraph.py: generate node data (targets and features) for real graphs 
4. make_dataset_realgraph_links.py: generate link data (targets and features) for real graphs 
5. config.py: location of project paths and data
6. utils.py: all helping functions to generate robustness scores for synthetic and real graphs
7. generate_egrscore.py: functions to generate egr and ws scores of different types of graph.
8. make_dataset_Bayesian_estgraph.py: generate target scores(egr or ws) for the estimated graph from gsp toolbox
9. make_dataset_Bayesian_obsgraph.py: generate embedding vector from observed graph for MAP estimate.


================ src.models files ====================

1.Semisup_Graphsage_nodebtw: main file for training algorithm for node prediction
2.Graphsage_Linkegr_prediction: main file for training algorithm for link prediction
3.Graphsage_test.py: testing the performance of trained node prediction algorithm.
4.test_GraphSAGElink.py: testing the performance of trained link prediction model  
5. Graphsage_SupvisedClassf.py: main file for training algorithm for node classification problem
6. Graphsage_Supervisclassf_Bayesian.py: main file for training algorithm for node classification in bayesian
setting.
7. GraphsageEmbmodel_Sup_Bayesian.py: main file for supervised mode of generating embeding for nodes from observed graph in bayesian setting
8. GraphsageEmbmodel_unSup_Bayesian.py: main file for unsupervised mode of generating embeding for nodes from observed graph in bayesian setting
9. MCdropout_MNIST.py: for simulating MC dropout algorithm on GNN.
10. Graphsage_test_Bayesian.py: testing the performance of trained bayesian node prediction algorithm. 

=============== src.visualize files =================

1. visualize.py: contains all helping functions for plotting and visualizing results of trained models.

============== src_bayesian files =======================
1. Semisup_nodepred_uncertpropag: main file for training algorithm for node prediction with uncertainty propag.