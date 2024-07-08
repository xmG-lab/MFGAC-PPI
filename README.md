# MFGAC-PPI
## System requirement
  python 3.7.7  
  numpy 1.19.1  
  pandas 1.1.0  
  torch 1.6.0  
  scikit-learn 1.0.2  
## Datasets
When processing protein sequences of pine wood nematode and pine tree, we have provided data files located in the ./data directory, which include 16 different protein sequences of pine wood nematode sourced from the PHI-base database. Additionally, we have supplied a file named interactions_data.txt that contains protein interaction data. It is important to note that all processed proteins are in PDB file format. We utilize the Alphafold2 tool for predicting and analyzing these protein sequences.
## Usage
  *residue_matrix.py: This script is used to extract residue matrices.
  *residue_feature.py: This script is used to extract residue features such as temperature factors and atomic weights.
  *ESM.py: This script is used to process protein sequences extracted from protein structures.
  *partition_dataset.py: This script is used to partition the dataset into balanced or unbalanced datasets, training set, and validation set.
  *run.py: This script is used for model training.
## Result
The trained model weights and the evaluation results using the test set are stored in the out_all directory, ready for direct model invocation.
Predictions for interactions between unknown pine wood nematode and pine tree protein pairs are saved in the predicted_results directory after calling the model.
