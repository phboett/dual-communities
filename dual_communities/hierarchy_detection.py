#!usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains methods to detect a hierarchical decomposition
of primal and dual graphs"""

import itertools


import networkx as nx
import numpy as np


def find_fiedler_vector(graph: nx.Graph, atol: float = 1e-12):
    """Get the eigenvector of the second eigenvalue of the graphs laplacian"""
    
    laplacian = nx.laplacian_matrix(graph).A
    
    ws, vv = np.linalg.eigh(laplacian)

    assert ws[1] > atol
    
    return ws[1], vv[:, 1]


def calc_fiedler_communities_dual_hierarchy_detection(G_dual,
                                                      G,
                                                      removed_edges,
                                                      return_dual_comms = False,
                                                      use_median = True,
                                                      clean_boundary_dublicate=False,
                                                      networkx_fiedler=True,
                                                      nx_fiedler_method='tracemin_pcg'):
    """Calc edges belonging to dual communities based on the Fiedler vector using graph that has
    been separated into two subgraphs, such that links might be missing (because the faces
    are no longer completely contained in the graph due to the removal of boundary edges)
    NOTE: in contrast to the other dual community detection methods, this method does not return
    the indices of the edges, but the actual edges"""
    
    #calculate the Fiedler vector
    if networkx_fiedler:
        v = nx.fiedler_vector(G_dual, method=nx_fiedler_method)
    else:
        _, v = find_fiedler_vector(G_dual)
        
    if use_median:
        median = np.median(v)
        print('Using median for communities')
    else:
        median = 0

    ###NOTE: this function assumes the dictionary mapping node ids of dual graph
    ###to edges forming faces in actual graph to be stored in the following place
    node_dict = G_dual.graph['node_dict_faces']
    nodes = list(G_dual.nodes())
    #### Communities expressed in terms of edges
    edges_community_1 = [node_dict[nodes[i]] for i in range(len(v)) if v[i] < median]
    edges_community_2 = [node_dict[nodes[i]] for i in range(len(v)) if v[i] > median]
    ##### NOTE: These lists of edges contain each edge twice: Once normal and once reversed
    ecomm1 = list(itertools.chain.from_iterable((edge, edge[::-1]) for i in range(len(edges_community_1))
                                      for edge in edges_community_1[i]
                                      if (edge not in removed_edges and edge[::-1] not in removed_edges)))
    ecomm2 = list(itertools.chain.from_iterable((edge, edge[::-1]) for i in range(len(edges_community_2))
                                      for edge in edges_community_2[i]
                                      if (edge not in removed_edges and edge[::-1] not in removed_edges)))
    boundary = list(set(ecomm1)-(set(ecomm1)-set(ecomm2)))
    ##now remove edges on the boundary from the two communities
    ecomm1 = list(set(ecomm1)-set(boundary))
    ecomm2 = list(set(ecomm2)-set(boundary))

    if clean_boundary_dublicate:
        boundary_set = set([tuple(sorted(xx)) for xx in boundary])
        boundary = list(boundary_set)
    
    if not return_dual_comms:
        return ecomm1,ecomm2,boundary

    else:
        dual_nodes_comm1 = [nodes[i] for i in range(len(v)) if v[i]<median]
        dual_nodes_comm2 = [nodes[i] for i in range(len(v)) if v[i]>median]

        return (ecomm1, ecomm2, boundary,
                dual_nodes_comm1, dual_nodes_comm2, v)


def calc_fiedler_communities_hierarchies(G, use_median = False,
                                          networkx_fiedler=True,
                                          nx_fiedler_method='tracemin_pcg'):
    """Calc edges belonging to communities based on the Fiedler vector"""
    
    #calculate the Fiedler vector
    if networkx_fiedler:
        v = nx.fiedler_vector(G, method=nx_fiedler_method)
    else:
        _, v = find_fiedler_vector(G)
        
    v = nx.fiedler_vector(G, method=nx_fiedler_method)

    if use_median:
        threshold = np.median(v)
        print('Using median for communities')
    else:
        threshold = 0
    index_dict = dict((value, idx) for idx,value in enumerate(list(G.nodes())))
    edges = list(G.edges())
    ecomm1 = []
    ecomm2 = []
    boundary = []
    for edge in edges:
        if v[index_dict[edge[0]]]>threshold and v[index_dict[edge[1]]]>threshold:
            ecomm1.append(edge)
        elif v[index_dict[edge[0]]]<threshold and v[index_dict[edge[1]]]<threshold:
            ecomm2.append(edge)
        else:
            boundary.append(edge)

    return ecomm1, ecomm2, boundary, v


def get_hierarchy_levels_dual(G,
                              G_dual,
                              nof_levels,
                              use_median = True,
                              clean_boundary_duplicates=False,
                              networkx_fiedler=True,
                              nx_fiedler_method='lobpcg',
                              return_fiedler_vecs=False):
    """Get hierarchical levels of graph based on dual communities

    Parameters
    ----------
    G : networkx Graph
        Graph for which hierarchical levels should be calculated
    G_dual : networkx Graph
        Dual Graph of G assumed to have attribute G_dual.graph['node_dict'] which is
        a dictionary containing for each node in G_dual the edges of the primal graph forming
        the corresponding facet
    nof_levels : int
        number of hierarchy levels to calculate

    Returns
    -------
    Graphs : list of networkx graphs
            A list of length nof_levels containing in its ith entry the Subgraphs resulting
            of the ith division of G into subgraphs, i.e. 2**i graphs at position i
    Duals : list of networkx graphs
            A list of length nof_levels containing in its ith entry the dual subgraphs resulting
            of the ith division of G_dual into subgraphs, i.e. 2**i graphs at position i
    boundaries : list of lists
            A list of length nof_levels containing at position i  the 2**(i-1) edges separating
            the subgraphs at the ith level
    """

    boundary_links = []
    Graphs = [[G]]
    Duals = [[G_dual]]
    removed_edges = []
    if return_fiedler_vecs:
        fiedler_vecs = []

    for i in range(nof_levels):
        boundary_links.append([])
        Graphs.append([])
        Duals.append([])
        if return_fiedler_vecs:
            fiedler_vecs.append([])
            
        for j in range(int(2**i)):
            ## following function calculates dual communities based on spectral clustering of Dual grah
            results = calc_fiedler_communities_dual_hierarchy_detection(Duals[i][j],
                                                                         Graphs[i][j],
                                                                         removed_edges,
                                                                         return_dual_comms = True,
                                                                         use_median = use_median,
                                                                         clean_boundary_dublicate=clean_boundary_duplicates,
                                                                         networkx_fiedler=networkx_fiedler,
                                                                         nx_fiedler_method=nx_fiedler_method)
            (edges_primal_community1, edges_primal_community2,
             boundary, dual_nodes_comm1, dual_nodes_comm2, fiedler_vec_r) = results
            
            if return_fiedler_vecs:
                fiedler_vecs[i].append(fiedler_vec_r)
                
            boundary_links[i].append(boundary)
            
            Graphs[i+1].append(Graphs[i][j].edge_subgraph(edges_primal_community1))
            Graphs[i+1].append(Graphs[i][j].edge_subgraph(edges_primal_community2))
            
            Duals[i+1].append(Duals[i][j].subgraph(dual_nodes_comm1))
            Duals[i+1].append(Duals[i][j].subgraph(dual_nodes_comm2))

        D = G.copy()
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for j in range(int(2**(i+1))):
            H = nx.compose(Graphs[i+1][j],H)
            
        D = nx.difference(D,H)
        removed_edges = list(D.edges())
        print('Finished level ' + str(i))
        #removed_edges = set(list(G.edges()))
        #for j in range(int(2**(i+1))):
        #    print('Checking Graph' +str(j))
        #    removed_edges -= set(list(Graphs[i+1][j].edges()))

    if return_fiedler_vecs:
        return Graphs, Duals, boundary_links, fiedler_vecs
    else:
        return Graphs, Duals, boundary_links


def get_hierarchy_levels_primal_graph(G,
                                      nof_levels,
                                      use_median = True,
                                      networkx_fiedler=True,
                                      nx_fiedler_method='lobpcg',
                                      return_fiedler_vecs=False):
    """Get hierarchical levels of graph based on primal communities

    Parameters
    ----------
    G : networkx Graph
        Graph for which hierarchical levels should be calculated
    nof_levels : int
        number of hierarchy levels to calculate

    Returns
    -------
    Graphs : list of networkx graphs
            A list of length nof_levels containing in its ith entry the Subgraphs resulting
            of the ith division of G into subgraphs, i.e. 2**i graphs at position i
    boundaries : list of lists
            A list of length nof_levels containing at position i  the 2**(i-1) edges separating
            the subgraphs at the ith level
    """

    boundary_links = []
    Graphs = [[G]]
    removed_edges = []
    
    if return_fiedler_vecs:
        fiedler_vecs = []

    for i in range(nof_levels):
        boundary_links.append([])
        Graphs.append([])
        if return_fiedler_vecs:
            fiedler_vecs.append([])
            
        for j in range(int(2**i)):
            ecomm1, ecomm2, boundary, fiedler_vec_r = calc_fiedler_communities_hierarchies(G = Graphs[i][j],
                                                                             use_median = use_median,
                                                                             networkx_fiedler=networkx_fiedler,
                                                                             nx_fiedler_method=nx_fiedler_method)
            if return_fiedler_vecs:
                fiedler_vecs[i].append(fiedler_vec_r)
                
            boundary_links[i].append(boundary)
            Graphs[i+1].append(Graphs[i][j].edge_subgraph(ecomm1))
            Graphs[i+1].append(Graphs[i][j].edge_subgraph(ecomm2))

        D = G.copy()
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        
        for j in range(int(2**(i+1))):
            H = nx.compose(Graphs[i+1][j],H)
            
        D = nx.difference(D, H)
        removed_edges = list(D.edges())
        print('Finished level ' +str(i))
    
    if return_fiedler_vecs:
        return Graphs, boundary_links, fiedler_vecs
    else:
        return Graphs, boundary_links


#### Following is for modifying graph
def remove_open_ends(G):
    """remove all open ends, i.e. nodes with a single neighbor from the graph
    and stop once no more single nodes are found"""
    
    found_single_node = 1
    node_dict = {}
    for i in range(len(list(G.nodes()))):
        node_dict[list(G.nodes())[i]]=i
    before = len(list(G.nodes()))
    removed_nodes = []
    while found_single_node:
        edges = list(G.edges())
        edges = np.array(edges).flatten()
        uni,count = np.unique(edges, return_counts=True)
        single_nodes = uni[np.where(count==1)]
        if not single_nodes.size:
            found_single_node =0
        else:
            removed_nodes.append(single_nodes)
            G.remove_nodes_from(single_nodes)

    ###flatten list but subtract 1, since indexing starts at 0
    #removed_indices=[val-1 for sublist in removed_nodes for val in sublist]
    kept_indices = [node_dict[val] for val in list(G.nodes())]
    after = len(list(G.nodes()))
    print("Removed " + str(before-after) + " single nodes from graph")

    return kept_indices, G
