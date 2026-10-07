# EGCPPIS
EGCPPIS is an end-to-end deep learning framework for identifying protein–protein interaction sites based on hierarchical graph representations and equivariant graph neural networks.


# Dependency
torch=2.1.1+cu118

torch-geometric=2.4.0

numpy=1.26.2

scikit-learn=1.6.1

einops=0.7.0

# embedding file
The preprocessed embedding file can be obtained from the following link：https://drive.google.com/drive/folders/1ahBAePK0waZJ82LR1NVMHx4mlS8zLowa


# Code Structure Description
To enhance reproducibility, we provide a detailed description of the files included in the GitHub repository:

ppi_main.py: The main entry file for training and testing the EGCPPIS model, including data loading, model initialization, and training pipeline.

Model_alpha.py: Defines the core architecture of the EGCPPIS model and its hierarchical graph representation modules.

egnn_pytorch.py: Implements the Equivariant Graph Neural Network (EGNN) used for atom-level feature extraction.

graphsage.py: Implements the GraphSAGE-based module used for residue-level feature aggregation.

evaluation.py: Contains performance evaluation functions, including accuracy, precision, recall, F1-score, and ROC-AUC calculations.

utils.py: Provides general utility functions such as data preprocessing, feature normalization, and graph construction.

utils_Test.py: Contains helper functions for inference and test-stage operations.

Test_31.py, Test_60.py, Test_315.py: Independent test scripts for different datasets or experimental configurations.

requirements.txt: Lists all dependencies and library versions required to reproduce the experiments.


