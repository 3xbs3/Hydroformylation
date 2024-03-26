# Hydroformylation
This repository was constructed to archive the work for the journal titled `Implementing Quantum Information and Machine Learning for Comprehensive Regioselectivity Prediction in Terminal Olefin Hydroformylation`.  
![]Figure.png
## Machine Learning
### Database 
The file 'Database.csv' serves as our database for the hydroformylation reaction data.
### Feature Engineering
#### Smiles2Structures
To implement Smiles2Structures in a Linux system, there are only a few simple steps required:  
`cd Example/automation_ligand-m`  
`qsub Automation.sh`  
You can locate the log file generated by Gaussian 16 in the relevant folder. 
#### Structures2Features
Please refer to the folder located at `Scripts/Feature`. 
### Graph Representation
Please refer to the file located at `Scripts/MPNN/MPNN.py`.
### Point Cloud Representation
Please refer to the file located at `Scripts/Pointnet/Pointnet.py`.
### Classic model
Please refer to the file located at `Scripts/Tradition/All-olefin.ipynb`. This applies to the other models as well.
