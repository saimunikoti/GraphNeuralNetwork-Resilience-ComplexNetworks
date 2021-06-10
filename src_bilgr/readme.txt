===================== Folder structure: BILGR for node classification ==============

1. src_bilgr/data/make_dataset_Bayesian_obsgraph.py : generates observation graph and target class for nodes

2. src_bilgr/models/GraphsageEmbmodel_Sup_Bayesian.py: Train supervised classification model with node   targets; generate embedding for nodes.

3. src_bilgr/models/make_dataset_Bayesian_estgraph.py: estimate graph from new adjacency matrix from GSP

4. utils.py: helping functions

5. src_bilgr/features/build_features.py: helping function for feature extyraction

6. src_bilgr/models/Graphsage_Supervisclassf_Bayesian.py: Train supervised classification model on the estimated graph for node final class predictions.

7. src_bilgr/visualization/visual.py: helping functions for visualizing features 

===============================================================================================================