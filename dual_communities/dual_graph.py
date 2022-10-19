#!/usr/bin/python3
# -*- coding: utf-8 -*

"""Functions needed to create dual graphs from graphs."""

import numpy as np
import networkx as nx

from numba import jit

from sage.graphs.all import Graph


#Following two functions are taken from https://plot.ly/python/polygon-area/
def polygon_sort(corners):
    n = len(corners)
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    cornersWithAngles = []
    for x, y in corners:
        an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        cornersWithAngles.append((x, y, an))
    cornersWithAngles.sort(key = lambda tup: tup[2])
    result = list(map(lambda  x: (x[0], x[1]), cornersWithAngles))
    return result


def polygon_area(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def determine_outer_face(faces, pos):
    """Calculate outer face by calculating face of maximal area."""
    
    current_area=0
    current_index=0
    
    for i in range(len(faces)):
        ##face is constructed of edges, so get corresponding nodes and their positions
        flat=[item for sublist in faces[i] for item in sublist]
        
        if type(faces[i][0][0])==type((1,2)):
            #need to handle nodes indexed by tuples differently
            flat_temp=np.unique(flat,axis=0)
            flat=[tuple(x) for x in flat_temp]
        else:
            flat=np.unique(flat)

        flat_pos=[pos[flat[j]] for j in range(len(flat))]
        corners_sorted = polygon_sort(flat_pos)
        area = polygon_area(corners_sorted)
        if area>=current_area:
            current_area=area
            current_index=i
            
    return current_index


def loop_for_dual_noloops(faces, outer_face, weightdict={}):
    """Extract loop from dual_graph function to this function to run it in numba.

    Args:
        faces (_type_): faces of graph
        outer_face (_type_): face sourounding the entire graph
        weightdict (dict, optional): ??. Defaults to {}.

    Returns:
        edges, node_ict: _description_
    """
    
    nodes = []
    #this establishes a mapping between nodes in dual graph
    # and faces in actual graph
    node_dict = {}
    edges = []
    #following is to prevent dictionary from given keyerror
    # when trying out reverted keys
    ww = weightdict.copy()
    
    for key in ww.keys():
        weightdict[key[::-1]] = weightdict[key]
        
    for i in range(len(faces)):
        if not i==outer_face:
            nodes.append(i)
            face = faces[i]
            node_dict[i] = face
            for j in range(i+1,len(faces)):
                face2 = faces[j]
                twisted_face2 = [e[::-1] for e in face2]
                ##check if faces are connected
                are_connected = False
                if not set(face).isdisjoint(face2):
                    shared = list(set(face)-(set(face)-set(face2)))
                    are_connected = True

                elif not set(face).isdisjoint(twisted_face2):
                    shared = list(set(face)-(set(face)-set(twisted_face2)))
                    are_connected = True
                else:
                    are_connected = False

                if are_connected:
                    ## Dual graph has a connecting edge for each
                    # edge shared between the two
                    if not j == outer_face:
                        if len(weightdict):
                            weight=[]
                            for edge in shared:
                                weight.append(1./(weightdict[edge]))
                            weight=np.sum(weight)
                            
                        else:
                            weight=len(shared)
                        edges.append((i, j, weight))

    return edges, node_dict

@jit
def get_dual_pos(faces,pos):
    """Get position of nodes in dual graph wrt positions in old graph"""
    
    dual_pos={}
    
    for i in range(len(faces)):
        average_pos=[(np.array(pos[edge[0]])+np.array(pos[edge[1]]))/2 for edge in faces[i]]
        average_pos=np.mean(average_pos,axis=0)
        dual_pos[i]=list(average_pos)
        
    return dual_pos


def dual_graph_noloops(weightdict, faces, outer_face):
    """Construct dual graph neglecting multiple connections
    between faces and connections to the outer face

    Args:
        weightdict (_type_): _description_
        faces (_type_): _description_
        outer_face (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    F = nx.Graph()
    edges, node_dict = loop_for_dual_noloops(faces, 
                                             outer_face,
                                             weightdict)
    F.add_nodes_from(node_dict.keys())
    F.add_weighted_edges_from(edges)
    F.graph['node_dict_faces'] = node_dict
    
    return F


def create_dual_from_graph(graph, min_component_size=None):
    """Create dual graph from given graph and nodes position by using the routines in  dual"""

    sage_graph = Graph(graph)
    faces = sage_graph.faces()
    
    outer_face = determine_outer_face(faces, pos=nx.get_node_attributes(graph, 'pos'))
    
    G_dual_noloops = dual_graph_noloops(nx.get_edge_attributes(graph, 'weight'),
                                             faces, outer_face=outer_face)
    graph.graph['faces'] = faces
    
    dual_pos = get_dual_pos(faces, pos=nx.get_node_attributes(graph, 'pos'))
    nx.set_node_attributes(G_dual_noloops, name='pos', values=dual_pos)
    
    # Remove small components:
    if min_component_size is not None:
        for comp in list(nx.connected_components(G_dual_noloops)):
            if len(comp) >= min_component_size:
                G_dual_noloops.remove_nodes_from(comp)
            else:
                pass
    
    return G_dual_noloops


def get_boundary_infos(graph: nx.Graph, boundaries, cut_lvl:int = 3):
    """get boundary links in primal graph corresponding to community 
    boundary in dual to visuals hierarchies"""
    
    partition_levels = np.arange(cut_lvl)
    boundary_dictionary = {(u, v): -1000 for u, v in graph.edges()}
    boundary_width = {(u, v): 1. for u, v in graph.edges()}
    
    for partition_level in partition_levels:
        
        for j in range(2**(partition_level)):
            
            for u, v in boundaries[partition_level][j]:
                boundary_dictionary[(u, v)] = partition_level
                boundary_width[(u, v)] = 3.

    return boundary_dictionary, boundary_width
