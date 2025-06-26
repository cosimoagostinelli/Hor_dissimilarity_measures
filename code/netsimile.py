"""
Graph distance based on:
Berlingerio, M., Koutra, D., Eliassi-Rad, T. & Faloutsos, C. 
NetSimile: A Scalable Approach to Size-Independent Network Similarity. arXiv (2012)

Code from netrd library (netrd: A library for network reconstruction and graph distances, McCabe et al., arXiv 2020)
"""

import networkx as nx
import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.stats import skew, kurtosis


def feature_extraction(G):
    """Node feature extraction.

    Parameters
    ----------
    G (nx.Graph): a networkx graph.
    ----------
    Returns:
        node_features (float): the Nx7 matrix of node features.
    """

    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 7))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]

    # clustering coefficient
    clusts = [node_clustering_dict[n] for n in node_list]

    # average degree of neighborhood
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # average clustering coefficient of neighborhood
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # number of edges in the neighborhood
    neighbor_edges = [
        egonets[n].number_of_edges() if node_degree_dict[n] > 0 else 0
        for n in node_list
    ]

    # number of outgoing edges from the neighborhood
    # the sum of neighborhood degrees = 2*(internal edges) + external edges
    # node_features[:,5] = node_features[:,0] * node_features[:,2] - 2*node_features[:,4]
    neighbor_outgoing_edges = [
        len(
            [
                edge
                for edge in set.union(*[set(G.edges(j)) for j in egonets[i].nodes])
                if not egonets[i].has_edge(*edge)
            ]
        )
        for i in node_list
    ]

    # number of neighbors of neighbors (not in neighborhood)
    neighbors_of_neighbors = [
        len(
            set([p for m in G.neighbors(n) for p in G.neighbors(m)])
            - set(G.neighbors(n))
            - set([n])
        )
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # assembling the features
    node_features[:, 0] = degs
    node_features[:, 1] = clusts
    node_features[:, 2] = neighbor_degs
    node_features[:, 3] = neighbor_clusts
    node_features[:, 4] = neighbor_edges
    node_features[:, 5] = neighbor_outgoing_edges
    node_features[:, 6] = neighbors_of_neighbors

    return np.nan_to_num(node_features)




def graph_signature(G):
    """
    Returns the signature vector of a graph (nx.Graph)
    as described in NetSimile: "A Scalable Approach to 
    Size-Independent Network Similarity", Berlingerio et al.,
    arXiv 2012.
    """

    node_features = feature_extraction(G)
    
    n_stats = 3     # replace 3 with 5 if skewness and kurtosis are included
    signature_vec = np.zeros(7 * n_stats)   

    # for each of the 7 features
    for k in range(7):
        # find the mean
        signature_vec[k * n_stats] = node_features[:, k].mean()
        # find the median
        signature_vec[k * n_stats + 1] = np.median(node_features[:, k])
        # find the std
        signature_vec[k * n_stats + 2] = node_features[:, k].std()
        # find the skew
        #signature_vec[k * n_stats + 3] = skew(node_features[:, k])
        # find the kurtosis
        #signature_vec[k * n_stats + 4] = kurtosis(node_features[:, k], fisher=False)

    return signature_vec






def weighted_graph_signature(G):
    """
    Step 1: extract the features of a weighted network.
    The features are, for each node i: (1) number of i's neighbors; 
    (2) i's strength; (3) i's clustering coefficient; (4) std of 
    weights of i's edges; (5) average number of neighbors' of
    i's neighbors; (6) average strength of i's neighbors; (7) number of 
    neighbors of i's egonet (i.e., number of nodes at distance 2 from i).
    The weighted clustering coefficient is the one defined in: 
    "The architecture of complex weighted networks", A. Barrat et al., PNAS (2004).

    Step 2: compute mean, median, and std of all the distributions listed above
    and concatenate the values in a feature vector representing the wighted
    network.
    """

    # 1 - list of number of neighbors of each node (i.e. degree)
    pd1_nneig = []
    
    # 2 - lists of weights of each node's edges, keyed by node IDs
    n_dict_ws = defaultdict(list)
    for e in G.edges:
        w = G.get_edge_data(e[0],e[1])['weight']
        n_dict_ws[e[0]].append(w)
        n_dict_ws[e[1]].append(w)

    # dictionry of strengths keyed by node IDs
    str_dict = {n: sum(x) for n,x in n_dict_ws.items()}
    # add isolated nodes
    for n in nx.isolates(G):
        str_dict[n] = 0
    pd2_str = list(str_dict.values())
    
    # 3 - list of nodes' clustering coefficient
    pd3_clst = []

    # 4 - std of weights of nodes' edges
    pd4_stdw = [np.std(x) for x in n_dict_ws.values()]
    # add isolated nodes
    pd4_stdw += [0.] * nx.number_of_isolates(G)

    # 5 - avg number of neighbors of each node
    pd5_neig_nneig = []

    # 6 - avg strenght of nodes' neighbors
    pd6_str_neig = []

    # 7 - number of neighbors of each nodes' egonet
    pd7_ego_neig = []
    
    
    for i in G.nodes:

        neig_i = list(G.neighbors(i))
        k_i = len(neig_i)
        pd1_nneig.append(k_i)

        # for isolated nodes
        if k_i==0:
            pd3_clst.append(0.)
            pd5_neig_nneig.append(0.)
            pd6_str_neig.append(0.)
            pd7_ego_neig.append(0.)
        else:
            nneig_neig_i = [len(list(G.neighbors(j))) for j in neig_i]
            pd5_neig_nneig.append( np.mean(nneig_neig_i) )
            pd6_str_neig.append( np.mean([str_dict[j] for j in neig_i]) )
            
            neig_neig_i = set()
            for j in neig_i:
                neig_j = set(G.neighbors(j))
                neig_neig_i = neig_neig_i.union(neig_j)
            # do not count i and its neighbors
            pd7_ego_neig.append( len( neig_neig_i-set(neig_i) ) - 1 )

            if k_i==1:
                pd3_clst.append(0.)
            else:
                cc = 0.
                for l,m in combinations(neig_i, 2):
                    if l in G.neighbors(m):
                        cc += G.get_edge_data(i,l)['weight'] 
                        cc += G.get_edge_data(i,m)['weight']
                cc /= (str_dict[i] * (k_i - 1))
                pd3_clst.append(cc)
                
    
    distr_list = [pd1_nneig, pd2_str, pd3_clst, pd4_stdw, 
                  pd5_neig_nneig, pd6_str_neig, pd7_ego_neig]
    g_vec = []
    for fd in distr_list:
        g_vec += [np.median(fd), np.mean(fd), np.std(fd)]
            
    return (g_vec)