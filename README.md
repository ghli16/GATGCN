Using graph attention network and graph convolutional network to explore human circRNA-disease associations based on multi-source data


## Code
### Environment Requirement
The code has been tested running under Python 3.6.8. The required packages are as follows:
- numpy == 1.15.4
- scipy == 1.1.0
- tensorflow == 1.12.0

Files: 
1.dataset
 1. circRNA_disease.csv stores known circRNA-disease association information;
 2. dis_sim stores disease fused similarity matrix;
 3. circRNA_sim stores circRNA fused similarity matrix
 4. circ_list and disease_list store circRNA ids and disease ids, respectively;
 
 
2.code      
  
1.main.py: the entrance of the program;  
2.model.py：the structure of GCN module；  
3.layers,py: the setting of GCN layers;    
4.bipartite_gat.py: the structure of GAT module;      
 

