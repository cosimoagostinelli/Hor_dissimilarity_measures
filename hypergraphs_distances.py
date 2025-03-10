import numpy as np
import networkx as nx
import xgi
from scipy.stats import kurtosis, skew
from scipy.cluster import hierarchy
from scipy.spatial import distance
from itertools import combinations


                  #  ------------------   Hyper-NetSimile  -------------------  #


def feature_vec (H):
    
    """""""""
    Returns a feature vector related to the given hypergraph.
    
    Input: A hypergraph H in xgi format (or a graph in networkx format).
    
    Output: a list of 40 values corrsponding to median, mean, std, skewness and kurtosis
            of the distributions of 8 structural (local) features of the hypergraph.
            There are 4 node-level features, namely: number of neighbors; hyperdegree; 
            hyperclustering coefficient; average hyperedge size.
            And 4 neighborhood-level (averaged) features: number of neighbors' neighbors;
            neighbors' hyperdegree; neighbors' hyperclustering coefficient; number of
            neighbors of the egonet of the selected node (i.e., number of nodes at
            distance 2 from i).
            
    """""""""
    deg_dict = H.nodes.degree.asdict()

                    # nodes' features

    #list of number of neighbors of each node
    pd1_nneig = []
    
    #list of hyperdegree of each node
    pd2_hdeg = list(deg_dict.values())
    
    #list of hyper clustering coefficient of each node
    clst_dict = xgi.local_clustering_coefficient(H)
    pd3_hclst = list(clst_dict.values())
    
    #list of average size of nodes' hyperedges
    pd4_avg_hsize = []
    
    #list of std of size of nodes' hyperedges
    pd5_std_hsize = []

                    # neighbors' features (averaged)
    
    #list of average neighbors' number of nieghbors
    pd6_neig_nneig = []

    #list of average neighbors' hyperdegree of each node
    pd7_neig_hdeg = H.nodes.average_neighbor_degree.aslist()

    #list of average neighbors' hyper clustering coefficient of each node
    pd8_neig_hclst = []

    #list of number of neighbors of each node's egonet
    pd9_ego_neig = []


    for i in H.nodes:

        neig_i = H.nodes.neighbors(i)
        pd1_nneig.append( len(neig_i) )

        # for isolated nodes
        if len(neig_i)==0:
            pd4_avg_hsize.append(0.)
            pd5_std_hsize.append(0.)
            pd6_neig_nneig.append(0.)
            pd8_neig_hclst.append(0.)
            pd9_ego_neig.append(0.)
        else:
            edge_neig_i = xgi.edge_neighborhood(H, i, include_self=True)
            hsizes = [len(j) for j in edge_neig_i]
            pd4_avg_hsize.append( np.mean(hsizes) )
            pd5_std_hsize.append( np.std(hsizes) )
            
            neig_nneig_i = [len(H.nodes.neighbors(j)) for j in neig_i]
            pd6_neig_nneig.append( np.mean(neig_nneig_i) )
            
            neig_clst = [clst_dict[j] for j in neig_i]
            pd8_neig_hclst.append( np.mean( neig_clst ) )

            neig_neig_i = set()
            for j in neig_i:
                neig_j = H.nodes.neighbors(j)
                neig_neig_i = neig_neig_i.union(neig_j) 
            # do not count i and its neighbors
            pd9_ego_neig.append(len(neig_neig_i-neig_i)-1)
        

    features_distr_list = [pd1_nneig, pd2_hdeg, pd3_hclst, pd4_avg_hsize, pd5_std_hsize,
                           pd6_neig_nneig, pd7_neig_hdeg, pd8_neig_hclst, pd9_ego_neig]
    hgraph_fvec = []

    for f_distr in features_distr_list:

        hgraph_fvec += [np.median(f_distr),
                     np.mean(f_distr),
                     np.std(f_distr),
                     skew(f_distr, bias=False),
                     kurtosis(f_distr, bias=False) ]
        
    return (hgraph_fvec)




                #  ------------------   Hyper-Portrait Divergence  -------------------  #
    
    
def H_to_G_mapping(H):
    
    """""""""""
    Map the hypergraph to a graph where each node represents
    a former hyperedge, and two nodes are connected if the original
    hyperedges shared at least one node. Every node has an attribute,
    that is the size of the hyperedge it represents.
    
    Parameters:
    ---------------
    
    H : a hypergraph in xgi format
    
    ---------------
    
    Returns: The network G resulting from the mapping (networkx). 
    
    """""""""""

    new_nodes = list(H.edges)
    new_edges = []
    sizes = H.edges.size.asdict()
    
    for (id1,id2) in combinations(H.edges, 2):
        e1 = H.edges.members(id1)
        e2 = H.edges.members(id2)
        if len(e1.intersection(e2)) > 0:
            new_edges.append((id1,id2))      
            
    G = nx.Graph()
    G.add_nodes_from(new_nodes)
    G.add_edges_from(new_edges)
    nx.set_node_attributes(G, sizes, name='size')
    
    return G



def edge_portrait(H):
    
    """""""""""
    The edge-portrait of the given hypergraph H.
    The edge portrait is a tensor with four indices B_{m,n,l,k} which entries
    give the number of edges of size m having k edges of size n at distance l.
    Two edges are at distance 1 if they share at least one node.
    
    Parameters:
    ---------------
    
    H : a hypergraph in xgi format
    
    ---------------
    
    Returns: the edge-portrait of H, as a 4-dimensional numpy array. 
        
    """""""""""
    
    G = H_to_G_mapping(H)
    sizes_dict = nx.get_node_attributes(G, 'size')
    s_max = np.max( xgi.unique_edge_sizes(H) )
    N = G.number_of_nodes()
    # connected components
    CC = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    dia = np.max( [nx.diameter(g) for g in CC] )
    B = np.zeros((s_max-1, s_max-1, dia+1, N), dtype=int) 
    
    for Gc in CC:
        for i in Gc.nodes:
            m = sizes_dict[i]-2
            dist_dict = nx.shortest_path_length(Gc, i)
            counter = np.zeros((s_max-1, dia+1), dtype=int)

            for j in Gc.nodes:
                counter[sizes_dict[j]-2][dist_dict[j]] += 1

            for n in range(s_max-1):
                for l in range(dia+1):
                    k = counter[n][l]
                    B[m][n][l][k] += 1 
                    
    return B



def pad_h_portraits (B1,B2):
    
    """""""""
    Make sure that two tensors are padded with zeros and/or trimmed of
    zeros in order to have the same dimensions.
    
    Parameters: 
    --------------
    
    B1,B2 : two edge-portraits of the hypergraphs to compare
    
    --------------
    
    Returns (B1,B2) : the two edge-portraits with same dimensions.
    
    """""""""
    
    # Bmats have N columns, find last occupied "column" and trim both down:
    lastcol1 = max(np.nonzero(B1)[3])
    lastcol2 = max(np.nonzero(B2)[3])
    lastcol = max(lastcol1,lastcol2)
    B1 = B1[:,:,:,:lastcol+1]
    B2 = B2[:,:,:,:lastcol+1]
    
    max_dims = [ max(B1.shape[i],B2.shape[i]) for i in range(4) ]
    
    for i in range(4):
    
        dims = list(B1.shape)
        dims[i] = max_dims[i]-dims[i]   
        to_stack = np.zeros(dims, dtype=int)
        B1 = np.append(B1, to_stack, axis=i)
        
        dims = list(B2.shape)
        dims[i] = max_dims[i]-dims[i]    
        to_stack = np.zeros(dims, dtype=int)
        B2 = np.append(B2, to_stack, axis=i)
      
    return (B1, B2)



def hyper_portrait_divergence(B1, B2):
    
    """""""""
    Dissimilarity measure between the two hypergraphs H1, H2, based on the
    generalization of the portrait divergence (Bagrow, 2019). It is defined as
    the Jensen-Shannon divergence between the distributions P1, P2 associated 
    to the hypergraphs. P is built upon the edge-portrait of the hypergraph
    (see edge_portrait(H) function): P(m,n,l,k) = B_{m,n,l,k}.
    
    Parameters :
    --------------
    B1, B2 : can be either the two hypergraphs to compare, in xgi format,
             or their edge-portraits, obtained via edge_portrait(H) function.
    --------------
    
    Returns : the hyper-portrait divergence between H1 and H2 (float).
        
    """""""""

    if isinstance(B1, xgi.Hypergraph):
        B1 = edge_portrait(B1)
        B2 = edge_portrait(B2)    
        
    B1, B2 = pad_h_portraits(B1,B2)
    P1 = np.ravel(B1)
    P2 = np.ravel(B2)
    JSD = distance.jensenshannon(P1,P2, base=2)
    
    return (JSD*JSD)






def DunnIndex(distances, n_clusters, method='single'):
    """
    Compute the Dunn's index DI for the given number of clusters. DI is computed
    as num/denom, where num is the minimum inter-cluster distance (i.e. the minimum
    distance between two points belonging to different clusters), and denom is the
    maximum intra-cluster distance (i.e. the maximium distance between two points
    belonging to the same cluster).
    The clusters are computed according to the scipy.cluster.hierarchy.linkage()
    function, with method='single'.
    
    Parameters :
    -------------
    distances : a 1-d array-like object containing the values of the upper
                triangular distance matrix, that is, the list of distances
                between all possible pairs of elements (see scipy.spatial.distance.pdist).
                
    n_clusters : number of clusters to group the elements.
    
    method : method to perform the hierarchical clustering (default is 'single').
             See the documentation of scipy.cluster.hierarchy.linkage for available
             options.
             
    ------------
    
    Returns : the Dunn's index (float).
    
    """
    # build distance matrix
    D = distance.squareform(distances)
    # assign elements to clusters
    Z = hierarchy.linkage(np.array(distances), method=method)
    clst_assign = hierarchy.cut_tree(Z, n_clusters=n_clusters)
    idxs = dict()
    denoms = []
    nums = []
    
    for clst in range(n_clusters):      
        idxs[clst] = [idx for idx in np.where(clst_assign==clst)[0]]
        # compute denominator
        if len(idxs[clst])>1:
            dists_in_c = [D[i,j] for i,j in combinations(idxs[clst], 2)]
            denoms.append( np.max(dists_in_c) )
       
    for c1,c2 in combinations(range(n_clusters),2):
        # compute numerator
        dists_out_c = [D[i,j] for i in idxs[c1] for j in idxs[c2]]
        nums.append( np.min(dists_out_c) )
        
    DI = np.min(nums) / np.max(denoms)

    return DI
