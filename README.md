The ablation experiment code is stored in the ablation experiment folder.



1.MSFTTmodel.py:FTT-Transformer model with multi-scale feedforward neural network.

2.FTTmodel.py:FTT-Transformer model without any modifications.

3.raft\_gat.py,raft\_gcn.py,raft\_mlp.py:Modeling the relationships between Raft consensus protocol nodes using GAT, GCN, and MLP respectively.

4.train\_fold.py:MSFTT/FTT was fused with three different graph modeling networks and subjected to 5-fold cross validation.

5.train\_ftt\_fold.py:MSFTT/FTT experiment without adding graph features.

6.train\_vanilla\_fold.py:Transformer was fused with three different graph modeling networks and subjected to 5-fold cross validation.

7.train\_VTfold.py:Transformer experiment without adding graph features.





The code compared with other methods is stored in the comparative experiment folder.



1.train\_gdm.py:Generate Diffusion Model (GDM) comparison.

2.train\_lstm.py:Long Short-Term Memory (LSTM) comparison.

3.train\_svr.py:Support Vector Regression (SVR) comparison.





The performance optimization experiment code is stored in the optimization experiment folder.



1.train\_hybrid.py:Model training code.

2.recommend\_block.py:Implementation of Optimal Block Recommendation Algorithm.

