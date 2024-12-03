# Mixture of Expert Graph Transformer
This is a pytorch geometric implementation of a Graph Transformer model based on the paper: A Generalization of Transformer Networks to Graphs (https://arxiv.org/abs/2012.09699) by Dwivedi & Bresson. 
The file contains:
1) MultiHeadAttention: Class for implementing multi head attention on graphs
2) GTLayer: Class for implementing a graph transformer layer
3)        : Class that implement the complete model

We also provide two notebooks:
1) Demo: Implement the training of the graph transformer model on mutag without Positional encoding 
2) DemoPE: Implement the training of the graph transformer model on mutag with Laplacian Positional Encoding
