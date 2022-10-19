#!usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx


def redefined_index(list_of_tuples: list, element: tuple) -> int:
    """redefine implemented index method for list 
    
    Parameters
    ----------
    list_of_tuples : list
        A list containing tuples
    element : tuple
        A single tuple whose position in the list is calculated ignoring tuple orientation
    
    Returns
    -------
    integer 
        index of element in list_of_tuples ignoring orientation of tuples
    """
    
    assert isinstance(list_of_tuples,list)
    assert isinstance(element,tuple)
    
    try:
        index = list_of_tuples.index(element)
    except ValueError:
        index = list_of_tuples.index(element[::-1])
        
    return index


def calc_PTDF_matrix(G: nx.Graph, I: np.ndarray = np.array([])):
    """Calculate Power Transfer Distribution Factor (PTDF) Matrix for power injections along graph's edges
    NOTE: This PTDF matrix is already multiplied from the right by the graphs 
    incidence matrix, thus assuming power injections only to take place at the terminal
    ends of edges and resulting in a nof_edges x nof_edges matrix
    
    Parameters
    ----------      
    G : networkx graph
         Graph based on which Laplacian matrix and thus PTDF matrix is calulated
    I : (optional) numpy array
         Represents oriented edge node incidence matrix 
    
    
    Returns
    -------
    numpy array
         Power Transfer Distribution Factor matrix of dimension number_of_edges x number_of_edges
    """
    
    B = nx.laplacian_matrix(G).todense()
    if not I.size:
        I = nx.incidence_matrix(G,oriented=True).todense()

    edges = [(list(G.nodes()).index(u),list(G.nodes()).index(v))
             for u,v in G.edges()]
    line_weights = np.array([B[e] for e in edges])
    B_d = -np.diag(line_weights)
    #multi_dot is supposed to find the most efficient way of performing the matrix multiplication
    #latter term is B inverse matrix
    try: 
        #implicit matrix inversion
        B_inv = np.linalg.solve(B, I)
        PTDF_matrix = np.linalg.multi_dot([B_d,I.T,B_inv])
        
        ### This sometimes results in Singular Matrix error
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B)
        PTDF_matrix = np.linalg.multi_dot([B_d,I.T,B_inv,I])
        
    return PTDF_matrix


def calc_LODF_matrix(G: nx.Graph):
    """B is the nodal susceptance matrix, i.e. the weighted graph Laplacian"""
    
    PTDF_mat = calc_PTDF_matrix(G)
    LODF_matrix = PTDF_mat/(1-np.diag(PTDF_mat))
    np.fill_diagonal(LODF_matrix, -1)
    #LODFs have to be mutually zero, but are sometimes calculated wrongly
    #LODF_matrix[np.isclose(LODF_matrix,0)+np.isclose(LODF_matrix,0).T]=0.0
    #LODFs on bridges are not well-defined, thus set them to nan
    
    if type(G) == type(nx.Graph()):
        bridges = nx.bridges(G)
    elif type(G) == type(nx.MultiGraph()):
        bridges = nx.bridges(nx.MultiGraph_to_Graph(G))
    else:
        raise NotImplementedError('not implemented for this kind of graph')
        
    bridges = [(list(G.edges()).index(e)) for e in bridges]
    for bridge in bridges:
        LODF_matrix[:, bridge] = np.nan

    return LODF_matrix


def shortest_edge_distance(G,e1,e2,weighted=False):
    """Calculate the edge distance between edge e1 and edge e2 by taking the minimum of all
    shortest paths between the nodes
    
    Parameters
    ----------      
    G : weighted or unweighted networkx graph
         Graph in which distance is to be calculated
    e1, e2 : tuples 
         Represent edges between which distance is to be calculated
    weighted : boolean
         If True, edge distance is calculated based on inverse edge weights
    
    Returns
    -------
    float 
         The length of the shortest path between e1 and e2 in G
    """

    assert isinstance(G,nx.Graph)
    assert isinstance(e1,tuple)
    assert isinstance(e2,tuple)
    
    possible_path_lengths = []
    F = G.copy()
    weight = nx.get_edge_attributes(F,'weight')
    if ((not len(weight)) or (not weighted)):
        weight = {e:1.0 for e in F.edges()}
        nx.set_edge_attributes(F,weight,'weight')
    inv_weight = {}
    for key in weight.keys():
        inv_weight[key] = 1/weight[key]
    nx.set_edge_attributes(F,inv_weight,'inv_weight')
   
    for i in range(2):
        
        for j in range(2):
            possible_path_lengths.append(nx.shortest_path_length(F, source=e1[i],
                                                                 target=e2[j], weight='inv_weight'))
            
    path_length = min(possible_path_lengths)+(F[e1[0]][e1[1]]['inv_weight']+F[e2[0]][e2[1]]['inv_weight'])/2
    
    return path_length  

def calc_flow_ratio(H: nx.Graph, F: nx.Graph, G:nx.Graph):
    """Calculate flow ratio for all possible trigger links and distances in Graph H
    from links in subgraph F of H to subgraph G in H and vice versa
    
    Parameters
    ----------      
    H : Networkx graph
       representing the whole graph
    G : Networkx graph
       subgraph of H representing one of the two modules of H
    F : Networkx graph
       subgraph of H representing the other module of H
    
    Returns
    -------
    numpy array
          contains distances [0] and flow ratios [1] evaluated
          for all possible trigger links
    """
    
    assert isinstance(H, nx.Graph)
    assert isinstance(F, nx.Graph)
    assert isinstance(G, nx.Graph)
    
    
    ### get the indices of all edges in F and in G
    edges_in_G_indices = [redefined_index(list_of_tuples=list(H.edges()),element=e) for e in G.edges()]
    edges_in_F_indices = [redefined_index(list_of_tuples=list(H.edges()),element=e) for e in F.edges()]
    
    ### calculate PTDF matrix for the given graph
    PTDF = calc_PTDF_matrix(H)
    flow_ratio = []
    distances = []
    
    ### Now iterate over all possible trigger links in G
    for trigger_index in edges_in_G_indices:
        trigger_link = list(H.edges())[trigger_index]

        ### calculate distance from trigger link to all possible other links
        edge_distance_uw = [shortest_edge_distance(G = H,e1 = e,e2 = trigger_link,weighted = False) for e in H.edges()]
        max_dist = np.max(edge_distance_uw)
        for d in np.arange(1,max_dist):
            ### get all edges at a distance d to the trigger link
            edges_distance_d_G = [i for i in edges_in_G_indices if edge_distance_uw[i]==d]
            edges_distance_d_F = [i for i in edges_in_F_indices if edge_distance_uw[i]==d]
            ### calculate the absolute ration of mean flows at this distance, if there are links in G and F at the given distance
            if len(edges_distance_d_G) and len(edges_distance_d_F):
                flow_ratio.append(np.round(np.mean(np.abs(PTDF[edges_distance_d_F,trigger_index])),decimals = 12)/np.round(np.mean(np.abs(PTDF[edges_distance_d_G,trigger_index])),decimals = 12))
                distances.append(d)
    ### Iterate over all possible trigger links in F
    for trigger_index in edges_in_F_indices:
        trigger_link = list(H.edges())[trigger_index]
        
        ### calculate distance from trigger link to all possible other links
        edge_distance_uw = [shortest_edge_distance(G = H,e1 = e,e2 = trigger_link,weighted = False) for e in H.edges()]
        max_dist = np.max(edge_distance_uw)
        for d in np.arange(1,max_dist):
            ### get all edges at a distance d to the trigger link
            edges_distance_d_G = [i for i in edges_in_G_indices if edge_distance_uw[i]==d]
            edges_distance_d_F = [i for i in edges_in_F_indices if edge_distance_uw[i]==d]
            ### calculate the absolute ration of mean flows at this distance, if there are links in G and F at the given distance
            if len(edges_distance_d_G) and len(edges_distance_d_F):
                flow_ratio.append(np.round(np.mean(np.abs(PTDF[edges_distance_d_G,trigger_index])),decimals = 12)/np.round(np.mean(np.abs(PTDF[edges_distance_d_F,trigger_index])),decimals = 12))
                distances.append(d)
    distances = np.array(distances)
    flow_ratio = np.array(flow_ratio)
    ratio_and_distances = np.array([distances,flow_ratio])
    return ratio_and_distances


def calc_flow_ratio_single_link(H: nx.Graph, F: nx.Graph, G: nx.Graph,
                                trigger_link: tuple):
    """Calculate flow ratio for a single trigger link
    
    Parameters
    ----------      
    H : Networkx graph
       representing the whole graph
    G : Networkx graph
       subgraph of H representing one of the two modules of H
    F : Networkx graph
       subgraph of H representing the other module of H
    trigger_link : tuple
       represents the edge that fails in order to calculate flow ratio
    
    Returns
    -------
    numpy array
         contains distances [0] and flow ratios [1] evaluated
         for all possible trigger links
    """
    
    assert isinstance(H,nx.Graph)
    assert isinstance(F,nx.Graph)
    assert isinstance(G,nx.Graph)
    assert isinstance(trigger_link,tuple)
        
    ### Check if link is located in module G or module F
    trigger_module = ''
    if G.has_edge(*trigger_link):
        trigger_module = 'G'
    elif F.has_edge(*trigger_link):
        trigger_module = 'F'
    else:
        print('Error, edge ' + str(trigger_link) + 'neither located in subgraph G, nor in subgraph F')
        return 
        
    
    ### get the indices of all edges in F and in G
    edges_in_G_indices = [redefined_index(list_of_tuples = list(H.edges()),element = e) for e in G.edges()]
    edges_in_F_indices = [redefined_index(list_of_tuples = list(H.edges()),element = e) for e in F.edges()]
    
    ### calculate PTDF matrix for the given graph
    PTDF = calc_PTDF_matrix(H)
    

    ### calculate distance from trigger link to all possible other links
    edge_distance_uw = [shortest_edge_distance(G = H,e1 = e,e2 = trigger_link,weighted = False) for e in H.edges()]
    max_dist = np.max(edge_distance_uw)
    
    flow_ratio = []
    distances = []
   
    trigger_index = redefined_index(list(H.edges()),trigger_link)

    for d in np.arange(1,max_dist):
        ### get all edges at a distance d to the trigger link
        edges_distance_d_G = [i for i in edges_in_G_indices if edge_distance_uw[i]==d]
        edges_distance_d_F = [i for i in edges_in_F_indices if edge_distance_uw[i]==d]
        ### calculate the absolute ratio of mean flows at this distance, if there are links in G and F at the given distance
        if len(edges_distance_d_G) and len(edges_distance_d_F):
            if trigger_module == 'G':
                flow_ratio.append(np.round(np.mean(np.abs(PTDF[edges_distance_d_F,trigger_index])),decimals = 12)/np.round(np.mean(np.abs(PTDF[edges_distance_d_G,trigger_index])),decimals = 12))
                
            elif trigger_module == 'F':
                flow_ratio.append(np.round(np.mean(np.abs(PTDF[edges_distance_d_G,trigger_index])),decimals = 12)/np.round(np.mean(np.abs(PTDF[edges_distance_d_F,trigger_index])),decimals = 12))
                
            distances.append(d)
            
    distances = np.array(distances)
    flow_ratio = np.array(flow_ratio)
    ratio_and_distances = np.array([distances,flow_ratio])
    
    return ratio_and_distances
