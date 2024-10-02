Code for analysis discussed in: Vaiva Vasiliauskaite, Nino Antulov-Fantulin. Generalization of Neural Network Models for Complex Network Dynamics. 

run_experiments_multi.py - Execution of this script will train a model (or an ensemble of models) defined in NeuralPsiBlock.py (Graph Neural Vector Field), to learn dynamics on a complex network. The model is trained on data where input is $\textbf{x}$ and output is generated using a vector field $F$. This code generates neural networks used in sections "Prediction accuracy of the trained models'' and "Identification of generalization limit'' of the paper.
run_experiments_neuralode.py - Used to train models using time series data. The results are discussed in section "Using noisy and irregularly sampled time series data".
run_experiments_gnns.py - Used to train models based on classic graph neural networks. The results are discussed in "Supplementary Note 4: Comparison with other graph neural networks''.
d_statistic.py - Includes functions to compute the statistical significance test.

---
Link to the preprint of the [paper](https://arxiv.org/abs/2301.04900).
