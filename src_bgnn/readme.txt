===================== Folder structure: BGNN for Node/link classification ==============

1. src_bgnn/data/make_dataset_uncertpropag.py : load graph and target vector with weights

2. src_bgnn/models/variance_trainedntwrk_uncertprop.py: get mean and variance of prediction from trained model weights and graph 

3. src_bgnn/models/Semisup_nodeclasf_uncertpropag.py: Train node classification model on weighted graphs 

4. utils.py: helping functions

5. src_bgnn/features/build_features.py: helping function for feature extyraction

6. src_bgnn/visualization/visual.py: helping functions for visualizing features 

===============================================================================================================