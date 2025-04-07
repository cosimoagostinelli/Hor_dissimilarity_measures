# Higher-order dissimilarity measures

Code to reproduce the results presented in the paper "Higher-order dissimilarity measures for hypergraph comparison", C. Agostinelli, M. Mancastroppa, A. Barrat (2025).

## Content

- `hypergraph_distances.py` contains the dissimilarity measures Hyper NetSimile and Hyperedge Portrait Divergence and the function to compute the Dunn Index.
- `hypergraph_models.py` contains the three generative models of hypergraphs used in the paper (ER, CM, WS).
- `hypergraph_null_models.py` contains the functions to generate null hypergraphs, either conserving the pairwise projection or randomizing the hyperedges.
- `netsimile.py` and portrait_divergence.py contain the functions to compute the pairwise dissimilarity measures. The code is taken from https://github.com/netsiphd/netrd/tree/master and https://github.com/bagrow/network-portrait-divergence respectively.
- notebooks 1-4 provide the results described in the paper; notebook 5 presents the figures and the values of the Rand and Dunn indices.



If you use this code, please cite the following paper: https://arxiv.org/abs/2503.16959



