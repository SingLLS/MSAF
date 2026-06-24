The ablation experiment code is stored in the ablation experiment folder.

1.MSFTTmodel.py:FTT-Transformer model with multi-scale feedforward neural network.
2.FTTmodel.py:FTT-Transformer model without any modifications.
3.raft_gat.py,raft_gcn.py,raft_mlp.py:Modeling the relationships between Raft consensus protocol nodes using GAT, GCN, and MLP respectively.
4.train_fold.py:MSFTT/FTT was fused with three different graph modeling networks and subjected to 5-fold cross validation.
5.train_ftt_fold.py:MSFTT/FTT experiment without adding graph features.
6.train_vanilla_fold.py:Transformer was fused with three different graph modeling networks and subjected to 5-fold cross validation.
7.train_VTfold.py:Transformer experiment without adding graph features.
8.train_loss.py:Loss function ablation.
9.train_csam.py:Fusion mechanism ablation.
10.train_rule_fold.py:Rule ablation.


The code compared with other methods is stored in the comparative experiment folder.

1.train_gdm.py:Generate Diffusion Model (GDM) comparison.
2.train_lstm.py:Long Short-Term Memory (LSTM) comparison.
3.train_svr.py:Support Vector Regression (SVR) comparison.
4.learningchain：https://github.com/JiShuWang/LearningChain


The performance optimization experiment code is stored in the optimization experiment folder.

1.train_hybrid.py:Model training code.
2.recommend_block.py:Implementation of Optimal Block Recommendation Algorithm.

