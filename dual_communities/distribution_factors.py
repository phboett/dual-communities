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